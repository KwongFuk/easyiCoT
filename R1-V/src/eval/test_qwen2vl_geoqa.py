from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
from math_verify import parse, verify
import time

def extract_answer(content):
    # 统一匹配三种前缀，查找后面第一个大写字母
    pattern = re.compile(r'(?:Answer:|The answer is|the answer is|### The final answer is:)\s*[^A-D]*([A-D])', re.IGNORECASE)
    match = pattern.search(content)
    if match:
        return match.group(1)
    return content


MODEL_PATH="/opt/data/private/others/cxy/models/Qwen2-VL-7B-Instruct" # qwen2vl model or grpoed model on geoqa train
BSZ=1 # reduce it if GPU OOM
OUTPUT_PATH="/opt/data/private/others/cxy/eval_outputs/cost_geoqa/Qwen2-VL-7B-Instruct_log.json"
PROMPT_PATH="./prompts/geoqa_test_prompts_new.jsonl"

#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

data = []
with open(PROMPT_PATH, "r") as f:
    for line in f:
        data.append(json.loads(line))


QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

REASONING_PROMPT_RATIONALE = """Provide a rationale to analyze the question.
Next, generate a step-by-step reasoning process to solve the problem. Ensure the steps are logical and concise.
Finally, provide a concise summary of the final answer in the following format: 'The final answer is: xxx.

Format your response with the following sections, separated by ###:
### Rationales:
### Let's think step by step.
### Step 1:
### Step 2:
...
### The final answer is: 

{Question}"""

# REASONING_PROMPT_COT = """You are given a math problem image, please solve the problem step by step.
# {Question}"""
REASONING_PROMPT_COT = """You are given a math problem image, please solve the problem step by step.
{Question}"""

messages = []

data = data

for i in data:
    message = [{
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": f"file://{i['image_path']}"
            },
            {
                "type": "text",
                "text": REASONING_PROMPT_COT.format(Question=i['question'])
            }
        ]
    }]
    messages.append(message)



torch.cuda.reset_peak_memory_stats()
all_outputs = []  # List to store all answers
mem_usage = []    # List to store memory usage after each batch

t1 = time.time()
# Process data in batches
for i in tqdm(range(0, len(messages), BSZ)):
    batch_messages = messages[i:i + BSZ]
    
    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    inputs.update({
            "interleave_inf": False,
            "selected_numbers": None,
            "num_added_tokens": None,
            "predicted_labels": None,
            "new_input_ids": None,
            "predicted_labels_output": None,
            "predict_threshold": 0.7,
            "interleave_sim": None,
            "interleave_cache": {},
            "return_dict_in_generate": True,
        })
    
    generated = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True)
    generated_ids = generated.sequences
    predicted_labels = generated.predicted_labels
    print(predicted_labels)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    all_outputs.extend(batch_output_text)

    # Record current GPU memory allocated
    current_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # in MB
    mem_usage.append(current_mem)

    print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")

max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
avg_mem = sum(mem_usage) / len(mem_usage)
t2 = time.time()

print(f"Total time taken: {t2 - t1:.2f} seconds")
print(f"Max GPU memory used: {max_mem:.2f} MB")
print(f"Average GPU memory used per batch: {avg_mem:.2f} MB")



final_output = []
correct_number = 0

for input_example, model_output in zip(data,all_outputs):
    model_answer = extract_answer(model_output)
    ground_truth = input_example['ground_truth']

    # Count correct answers
    if model_answer is not None and model_answer == ground_truth:
        correct_number += 1
        is_correct = True
    else:
        is_correct = False
    
    try:
        result = {
            'question': input_example,
            'ground_truth': ground_truth,
            'model_output': model_output,
            'extracted_answer':str(model_answer[0]) if model_answer is not None else None,
            'is_correct':is_correct
        }

    except Exception as e:
        print("no answer parsed",e,model_answer)
        result = {
            'question': input_example,
            'ground_truth': ground_truth,
            'model_output': model_output,
            'extracted_answer':None,
            'is_correct':is_correct
        }



    final_output.append(result)


# Calculate and print accuracy
accuracy = correct_number / len(data) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

# Save results to a JSON file
output_path = OUTPUT_PATH
with open(output_path, "w") as f:
    json.dump({
        'accuracy': accuracy,
        'time': t2 - t1,
        'average_time': (t2 - t1) / len(data),
        'max_memory': max_mem,
        'average_memory': avg_mem,
        'results': final_output
    }, f, indent=2, ensure_ascii=False)

print(f"Results saved to {output_path}")





