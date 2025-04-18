import json

file_path = '/mnt/petrelfs/share_data/suzhaochen/datasets/reachqa_final/reachqa_test.json'
already_processed_path = '/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/test_reachqa_result/gemini/ckpt/reachqa-test_gemini.jsonl'
with open(file_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)
process_data = set()
with open(already_processed_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        process_data.add(data['results']['results']['meta_data']['text'])
print(len(dataset))
selected_dataset = []
for data in dataset:
    if data['question'] not in process_data:
        selected_dataset.append(data)
print(len(selected_dataset))
# print(process_data)