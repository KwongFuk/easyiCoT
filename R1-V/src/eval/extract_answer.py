import re
import json
from math_verify import parse, verify
import ipdb

def extract_answer(content):
    # 统一匹配三种前缀，查找后面第一个大写字母
    pattern = re.compile(r'(?:Answer:|The answer is|the answer is)\s*[^A-D]*([A-D])', re.IGNORECASE)
    match = pattern.search(content)
    if match:
        return match.group(1)
    
    boxed_pattern = re.compile(r'\\boxed\{(.*?)\}')
    boxed_match = boxed_pattern.search(content)
    if boxed_match:
        return boxed_match.group(1).strip()

    # 如果都没有匹配，返回原始内容
    return content

with open("/opt/data/private/others/cxy/eval_outputs/geoqa/Qwen2.5-VL-7B-Instruct_log.json", "r") as f:
    datas = json.load(f)

correct_number = 0
for data in datas["results"]:
    model_output = data["model_output"]
    ground_truth = data["ground_truth"]
    model_answer = extract_answer(model_output)
    # ipdb.set_trace()
    if model_answer is not None and model_answer == ground_truth:
        correct_number += 1
        is_correct = True
    else:
        is_correct = False
    
acc = correct_number / len(datas["results"]) * 100 
print(f"Accuracy: {acc:.4f}")

