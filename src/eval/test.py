from tqdm import tqdm
import re
from math_verify import parse, verify
import json

# Load data from JSON file
input_path = "/mnt/petrelfs/share_data/suzhaochen/datasets/chartgemma_cot/evaluate/qwen72b-train.json"
with open(input_path, "r") as f:
    data = json.load(f)
breakpoint()
# data = []
# with open(input_path, "r") as f:
#     for line in f:
#         line = line.strip()
#         if line:  # Skip empty lines
#             try:
#                 data.append(json.loads(line))
#             except json.JSONDecodeError:
#                 print(f"Skipping invalid JSON line: {line}")

correct_number = 0
final_output = []

def new_func(original_output):
    match = re.search(r'<answer>(\d+(\.\d+)?)</answer>', original_output)
    if not match:
        match = re.search(r'<answer>(.*?)</answer>', original_output)
    return match

for item in tqdm(data):
    original_output = item['qwen72b_label']
    ground_truth = item['label']
    
    # Extract the answer using regex
    match = new_func(ground_truth)
    if match:
        ground_truth = match.group(1)
    else:
        ground_truth = None
    model_answer = original_output
    # Count correct answers
    if model_answer is not None and float(verify(model_answer, ground_truth)) > 0:
        correct_number += 1
        is_correct = True
    else:
        is_correct = False
    if not is_correct:
        final_output.append({
            "image_path": item['image_path'],
            "query": item['question'],
            'label': item['label'],
            'predicted': model_answer,
            'is_correct': is_correct
        })

# Calculate and print accuracy
accuracy = correct_number / len(data) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

# Save the final output to a JSON file
output_path = "/mnt/petrelfs/share_data/suzhaochen/datasets/chartgemma_cot/evaluate/qwen72b-train-final.json"
with open(output_path, "w") as f:
    json.dump(final_output, f, indent=4)
