cd R1-V/src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./log/Qwen2-VL-7B_interleaved_num4_54k_resacle50.txt" # Log file path

# CUDA_VISIBLE_DEVICES="7" torchrun --nproc_per_node="1" \
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir ../../../saves/grpo_qwen2vl_interleaved \
    --model_name_or_path ../../../saves/qwen2vl_7b_full_sft_interleaved/full/sft_epoch6_lr1e-6_warm01 \
    --dataset_name data \
    --deepspeed local_scripts/zero3_offload.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 200704 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-7B_interleaved_num4_54k_resacle50 \
    --save_steps 100 \
    --save_only_model True \
    --num_generations 4   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
