# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

import copy
import math
from PIL import Image
import io
from torch.nn.utils.rnn import pad_sequence

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model_id)
            pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.pad_token_id = pad_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            if not "Aria" in model_id:
                processing_class.image_processor.max_pixels = max_pixels
                processing_class.image_processor.min_pixels = min_pixels

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1, # HACK
            # num_return_sequences=self.num_generations,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
            # use_cache=False,
        )
        self.beta = args.beta
        self.interleave_token_id = 151650 # <|quad_start|>
        self.selected_token_id = 151651 # <|quad_end|>
        self.vis_start_token_id = 151652 # <|vision_start|>
        self.vis_end_token_id = 151653 # <|vision_end|>

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    # Get the per-token log probabilities for the completions for the model and the reference model
    # interleave mode
    def _get_per_token_logps_interleave(self, model, input_ids, attention_mask, pixel_values, image_grid_thw, predicted_labels):
        output = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw, selected_numbers = predicted_labels)
        logits = output.logits  # (B, L, V)
        predicted_logps = output.predicted_logps # 
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred 
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        per_token_logps = torch.stack(per_token_logps)
        
        # interleave logits
        # num_selected = predicted_labels.sum().item()
        # if num_selected == 0:
        #     predicted_logps_select = torch.empty((0,), device=logits.device, dtype=logits.dtype) # avoid deadlock
        # else:
        #     predicted_logps_select = predicted_logps.masked_select(predicted_labels).contiguous()
        # predicted_logps_full = torch.zeros_like(per_token_logps)
        # predicted_logps_full = predicted_logps_full.masked_scatter(
        #     selected_mask,
        #     predicted_logps_select
        # )
        # per_token_logps = torch.where(
        #     selected_mask,
        #     predicted_logps_full,
        #     per_token_logps
        # )
        
        return per_token_logps, predicted_logps
    
    def interleave_mask(self, completion_mask, completion_ids):
        # compute interleave logits
        # interleave_mask = ~(completion_ids == self.interleave_token_id) 
        # vis_start_mask = ~(completion_ids == self.vis_start_token_id)
        # vis_end_mask = ~(completion_ids == self.vis_end_token_id)
        
        # # vis_start_mask 向右扩展一位
        # vis_start_mask_right1 = torch.cat([torch.ones_like(vis_start_mask[:, :1]), vis_start_mask[:, :-1]], dim=1)
        # vis_start_mask = vis_start_mask & vis_start_mask_right1
        
        # # vis_end_mask 向右移一位
        # vis_end_mask = torch.cat([torch.ones_like(vis_end_mask[:, :1]), vis_end_mask[:, :-1]], dim=1)
        
        # interleave_completion_mask = completion_mask & interleave_mask & vis_start_mask & vis_end_mask
        
        
        # original version
        padding_mask = ~(completion_ids == self.generation_config.pad_token_id)
        interleave_mask = ~(completion_ids == self.interleave_token_id)
        selected_tokens_mask = ~(completion_ids == self.selected_token_id)
            
        interleave_mask_right1 = torch.cat([torch.ones_like(interleave_mask[:, :1]), interleave_mask[:, :-1]], dim=1)
        interleave_mask_right2 = torch.cat([torch.ones_like(interleave_mask_right1[:, :1]), interleave_mask_right1[:, :-1]], dim=1)
        interleave_mask_right3 = torch.cat([torch.ones_like(interleave_mask_right2[:, :1]), interleave_mask_right2[:, :-1]], dim=1)
        interleave_mask = interleave_mask & interleave_mask_right1 & interleave_mask_right2 & interleave_mask_right3
        
        selected_tokens_mask_right1 = torch.cat([torch.ones_like(selected_tokens_mask[:, :1]), selected_tokens_mask[:, :-1]], dim=1)
        selected_tokens_mask_right2 = torch.cat([torch.ones_like(selected_tokens_mask_right1[:, :1]), selected_tokens_mask_right1[:, :-1]], dim=1)
        selected_tokens_mask = selected_tokens_mask & selected_tokens_mask_right1 & selected_tokens_mask_right2
        
        interleave_completion_mask = completion_mask & interleave_mask & selected_tokens_mask & padding_mask
        return interleave_completion_mask
            
    def expand_zeros(mask, num_expand):
        expanded = mask.clone()
        for _ in range(num_expand):
            is_zero = (expanded == 0)
            # 检查右侧是否是非零或边界
            right_non_zero = torch.cat([is_zero[:, 1:], torch.zeros_like(is_zero[:, :1])], dim=1) == 0
            end_positions = is_zero & right_non_zero
            # 将结束位置右边的位置设为0
            expanded[:, 1:] = expanded[:, 1:] & ~end_positions[:, :-1]
        return expanded

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs
    
    def smart_resize(
        self, height: int, width: int, factor: int, min_pixels=256 * 28 * 28,max_pixels=1280 * 28 * 28,
    ):
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar
    
    def map_patch_numbers_batch(
        self,
        select_numbers_list,  # List of lists, each contains selected patch numbers for a sample
        h_bar_old_list,      # List of original heights after smart_resize
        w_bar_old_list,      # List of original widths after smart_resize
        h_bar_new_list,      # List of new heights after smart_resize
        w_bar_new_list,      # List of new widths after smart_resize
        old_merge=6,         # Original merge size (6 for 14x6 patches)
        new_merge=2,         # New merge size (2 for 14x2 patches)
        patch_size=14        # Patch size (default 14)
    ):
        """
        Maps selected patch numbers from the original patches (merge=6) to the new patches (merge=2) considering smart resizing.
        Processes each sample in the batch independently.
        """
        new_select_numbers = []
        
        for idx in range(len(select_numbers_list)):
            select_numbers = select_numbers_list[idx]
            h_bar_old = h_bar_old_list[idx]
            w_bar_old = w_bar_old_list[idx]
            h_bar_new = h_bar_new_list[idx]
            w_bar_new = w_bar_new_list[idx]
            
            # Original block parameters
            S_old = patch_size * old_merge  # 14*6=84
            merge_w_old = w_bar_old // S_old
            merge_h_old = h_bar_old // S_old
            
            # New block parameters
            S_new = patch_size * new_merge  # 14*2=28
            merge_w_new = w_bar_new // S_new
            merge_h_new = h_bar_new // S_new
            
            step_nums = []
            for nums in select_numbers:
                mapped_numbers = set()
                for num in nums:
                    # Calculate original block position
                    i = num // merge_w_old
                    j = num % merge_w_old
                    
                    # Original block coordinates (left-top to right-bottom)
                    x1_old = j * S_old
                    y1_old = i * S_old
                    x2_old = x1_old + S_old
                    y2_old = y1_old + S_old
                    
                    # Scale coordinates to new dimensions
                    x_scale = w_bar_new / w_bar_old
                    y_scale = h_bar_new / h_bar_old
                    
                    x1_new = x1_old * x_scale
                    y1_new = y1_old * y_scale
                    x2_new = x2_old * x_scale
                    y2_new = y2_old * y_scale
                    
                    # Determine new block indices
                    k_start = max(int(x1_new // S_new), 0)
                    k_end = min(int(math.floor((x2_new - 1e-9) / S_new)), merge_w_new - 1)
                    l_start = max(int(y1_new // S_new), 0)
                    l_end = min(int(math.floor((y2_new - 1e-9) / S_new)), merge_h_new - 1)
                    
                    # Collect all new blocks covered by the original block
                    for l in range(l_start, l_end + 1):
                        for k in range(k_start, k_end + 1):
                            new_num = l * merge_w_new + k
                            mapped_numbers.add(new_num)
                            
                step_nums.append(sorted(list(mapped_numbers)))
            new_select_numbers.append(step_nums)
        
        return new_select_numbers
    
    def interleave_prompts(self, prompts, images):
        merge_size = self.processing_class.image_processor.merge_size
        patch_size = self.processing_class.image_processor.patch_size
        mm_inputs = {} 
        mm_inputs.update(self.processing_class.image_processor(images, return_tensors="pt"))
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        
        if "<interleave>" in prompts:
            self.interleave_token = '<|quad_start|>'
            self.selected_token = '<|quad_end|>'
            
            if prompts.count("<interleave>") % 2 != 0:
                raise ValueError(f"The number of interleave placeholder must be an even number.")
            # import ipdb; ipdb.set_trace()
            content = prompts.split("<interleave>")
            content_list = content[::2]
            interleave_list = content[1::2]
            
            # select_numbers
            select_numbers = []
            for interleave in interleave_list:
                if interleave:
                    select_numbers.append([int(i.strip()) for i in interleave.split(",")])
                else:
                    select_numbers.append([])
            select_numbers = [select_numbers] # for only one image
            
            orig_patch_size = 14
            orig_merge_size = 6
            factor = orig_patch_size * orig_merge_size
            resized_heights = []
            resized_widths = []
            new_image_heights = []
            new_image_widths = []
            for i, img in enumerate(images):
                orig_width, orig_height = img.size
                resized_height, resized_width = self.smart_resize(
                    orig_height,
                    orig_width,
                    factor,
                    min_pixels=256 * 28 * 28,
                    max_pixels=1280 * 28 * 28,
                )
                resized_heights.append(resized_height)
                resized_widths.append(resized_width)
                new_image_height = patch_size * int(image_grid_thw[i][1])
                new_image_width = patch_size * int(image_grid_thw[i][2])
                new_image_heights.append(new_image_height)
                new_image_widths.append(new_image_width)
            
            # import ipdb; ipdb.set_trace()
            new_selected_numbers = self.map_patch_numbers_batch(
                select_numbers,
                resized_heights,
                resized_widths,
                new_image_heights,
                new_image_widths,
                orig_merge_size,
                new_merge=merge_size,
                patch_size=patch_size
            )
            
            interleaved_content = []
            for i, text in enumerate(content_list):
                interleaved_content.append(text)
                if i < len(interleave_list):
                    interleaved_content.append(self.interleave_token)
                    interleaved_content.append(f"<|vision_start|>{self.selected_token * len(new_selected_numbers[0][i])}<|vision_end|>")
                    # interleaved_content.append(self.selected_token * len(new_selected_numbers[0][i]))
            content = ''.join(interleaved_content)

        return content, new_selected_numbers
    
    def process_selected_tokens(self, prompt_completion_ids, predicted_labels):
        n_selected_tokens = (prompt_completion_ids == self.selected_token_id).sum().item()
        if predicted_labels is None:
            n_selected_numbers = 0
        else:   
            n_selected_numbers = (predicted_labels == 1).sum().item()
            i = predicted_labels.shape[0]-1
            while n_selected_numbers > n_selected_tokens and i >= 0:
                predicted_labels[i,:] = 0
                i -= 1
                n_selected_numbers = (predicted_labels == 1).sum().item()

        delta = n_selected_tokens - n_selected_numbers
        if delta > 0:
            selected_indices = torch.where(prompt_completion_ids == self.selected_token_id)[0]
            if len(selected_indices) >= delta:
                indices_to_remove = selected_indices[-delta:]
                all_indices = torch.arange(prompt_completion_ids.size(0), device=prompt_completion_ids.device)
                indices_to_keep = all_indices[~torch.isin(all_indices, indices_to_remove)]
                prompt_completion_ids = prompt_completion_ids[indices_to_keep]
            else:
                prompt_completion_ids = prompt_completion_ids[prompt_completion_ids != self.selected_token_id]
                
        return prompt_completion_ids, predicted_labels

    def process_interleave_tokens(self, prompt_completion_ids, predicted_labels):
        n_interleave_tokens = (prompt_completion_ids == self.interleave_token_id).sum().item()
        if predicted_labels is None:
            n_interleave_numbers = 0
        else:   
            n_interleave_numbers = predicted_labels.shape[0]
        
        if n_interleave_tokens > n_interleave_numbers:
            delta = n_interleave_tokens - n_interleave_numbers
            # 找到所有 interleave token 的位置
            interleave_indices = torch.where(prompt_completion_ids == self.interleave_token_id)[0]
            if len(interleave_indices) >= delta:
                cutoff_index = interleave_indices[-delta]  # 截断位置的起点
                prompt_completion_ids = prompt_completion_ids[:cutoff_index]
            else:
                # 如果数量不够，也就全删掉了
                prompt_completion_ids = prompt_completion_ids[prompt_completion_ids != self.interleave_token_id]
            
        elif n_interleave_tokens < n_interleave_numbers:
            # 处理预测标签多余的情况
            delta = n_interleave_numbers - n_interleave_tokens
            predicted_labels = predicted_labels[:-delta,:]  # 删掉最后 delta 个
        
        return prompt_completion_ids, predicted_labels
            

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        images = [x["image"] for x in inputs]
        images = [Image.open(io.BytesIO(image["bytes"])) for image in images]
        # prompts_text, selected_numbers = self.interleave_prompts(prompts_text, images)
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]
        image_grid_thw = prompt_inputs["image_grid_thw"]
        
        prompt_inputs["interleave_inf"] = True
        prompt_inputs["selected_numbers"] = None
        prompt_inputs["num_added_tokens"] = None
        prompt_inputs["predicted_labels"] = None
        prompt_inputs["new_input_ids"] = None
        prompt_inputs["predicted_labels_output"] = None
        prompt_inputs["predict_threshold"] = 0.7
        prompt_inputs["interleave_sim"] = None
        prompt_inputs["interleave_cache"] = {}
        prompt_inputs["return_dict_in_generate"] = True
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids_list = []
            predicted_labels_list = []
            interleave_token_num_idx_list = []
            for iter in range(self.num_generations):
                outputs_iter = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
                prompt_completion_ids_iter = outputs_iter.sequences.squeeze(0)
                predicted_labels_iter = outputs_iter.predicted_labels
                # filter prompt_completion_ids_iter
                prompt_completion_ids_iter, predicted_labels_iter = self.process_interleave_tokens(prompt_completion_ids_iter, predicted_labels_iter)
                prompt_completion_ids_iter, predicted_labels_iter = self.process_selected_tokens(prompt_completion_ids_iter, predicted_labels_iter)
                
                prompt_completion_ids_list.append(prompt_completion_ids_iter)
                # pad the predicted labels to [interleave_steps, img_embeds_len * num_generations]
                if predicted_labels_iter is not None:
                    predicted_labels_pad = torch.zeros(predicted_labels_iter.shape[0], predicted_labels_iter.shape[1]*self.num_generations, dtype=predicted_labels_iter.dtype, device=predicted_labels_iter.device)
                    predicted_labels_pad[:, iter*predicted_labels_iter.shape[1]:(1+iter)*predicted_labels_iter.shape[1]] = predicted_labels_iter
                    predicted_labels_list.append(predicted_labels_pad)
                    interleave_token_num_idx_list.append(predicted_labels_iter.shape[0])
                else:
                    interleave_token_num_idx_list.append(0)
                    continue
            prompt_completion_ids = pad_sequence(prompt_completion_ids_list, batch_first=True, padding_value=self.generation_config.pad_token_id)
            if len(predicted_labels_list) > 0:
                predicted_labels = torch.cat(predicted_labels_list, dim=0)
            else:
                predicted_labels = None
            # prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # interleave  
        interleave_completion_mask = self.interleave_mask(completion_mask, completion_ids)
        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1)
        image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)
        try:
            per_token_logps, interleave_logps = self._get_per_token_logps_interleave(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw, predicted_labels)
        except Exception as e:
            print(f"Error in _get_per_token_logps_interleave: {e}")
            print("pixel_values:", pixel_values)
            print("image_grid_thw:", image_grid_thw)
            print("prompt_completion_ids:", prompt_completion_ids)
            print("prompts:", prompts)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps, ref_interleave_logps = self._get_per_token_logps_interleave(self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw, predicted_labels)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps, ref_interleave_logps = self._get_per_token_logps_interleave(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw, predicted_labels)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        if interleave_logps is not None and ref_interleave_logps is not None:
            interleave_kl = torch.exp(ref_interleave_logps - interleave_logps) - (ref_interleave_logps - interleave_logps) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * interleave_completion_mask).sum(dim=1) / interleave_completion_mask.sum(dim=1)).mean()
        # loss = 0
        if predicted_labels is not None and interleave_logps is not None and ref_interleave_logps is not None:
            eps = 1e-8
            interleave_advantages = advantages.unsqueeze(1)
            repeated_advantages_list = []
            for i, repeat_times in enumerate(interleave_token_num_idx_list):
                advantage_row = interleave_advantages[i]  # shape [1]
                repeated_rows = advantage_row.repeat(repeat_times, 1)
                repeated_advantages_list.append(repeated_rows)
            interleave_advantages = torch.cat(repeated_advantages_list, dim=0)
            interleave_loss = torch.exp(interleave_logps - interleave_logps.detach()) * interleave_advantages
            interleave_loss = -(interleave_loss - self.beta * interleave_kl)
            # loss_2 = ((interleave_loss * predicted_labels).sum(dim=1) / (predicted_labels.sum(dim=1)+ eps)).mean()
            
            split_losses = torch.split(interleave_loss, interleave_token_num_idx_list, dim=0)
            split_predicted_labels = torch.split(predicted_labels, interleave_token_num_idx_list, dim=0)
            loss_2_list = []
            for cur_loss, cur_mask in zip(split_losses, split_predicted_labels):
                if cur_loss.shape[0] > 0:
                    loss_2 = ((cur_loss*cur_mask).sum(dim=1) / (cur_mask.sum(dim=1)+ eps)).mean()
                    loss_2_list.append(loss_2)
            if len(loss_2_list) > 0:
                loss_2 = torch.stack(loss_2_list).mean() / 50
                print("loss_2:", loss_2)
                loss += loss_2 
        
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # Log the metrics
        
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * interleave_completion_mask).sum(dim=1) / interleave_completion_mask.sum(dim=1)).mean()
        # mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
