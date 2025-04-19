from math_verify import parse, verify
import json
import re

def evaluate_metric(label, prediction):
    pattern = r'"actions": \[\{"name": "Terminate", "arguments": \{"ans": (.*?)\}'
    matches = re.findall(pattern, prediction)
    if len(matches) == 1:
        prediction = matches[0]
    else:
        return 0
    gold = parse('${0}$'.format(label))
    pred = parse('${0}$'.format(prediction))
    acc = 0
    if verify(gold, pred):
        acc+=1
    elif '%' in prediction[-1]:
        pred = prediction[:-1:]
        if '.' in pred:
            pred = pred.split('.')[0]
        if pred == label:
            acc+=1
    elif '.' in prediction:
        pred = prediction.split('.')[0]
        if pred == label:
            acc+=1
    return acc

ground_truth_path = '/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/test_dataset/chartgemma200.json'
# processed_path = '/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/use_tool_result/chartgemma/chartgemma_grpo_200.jsonl'
# processed_path = '/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/use_tool_result/chartgemma/chartgemma_4o.json'
processed_path = '/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/use_tool_result/chartgemma/chartgemma_grpo_200.jsonl'
# ground_truth_path = '/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/test_dataset/reachqa200.json'
# processed_path = '/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/use_tool_result/reachqa/reachqa_4o.json'

ground_truth_dict = dict()
with open(ground_truth_path, 'r', encoding='utf-8') as f:
    all_data = json.load(f)
for data in all_data:
    sol_match = re.search(r'<answer>(.*?)</answer>', data['label'])
    ground_truth_dict[data['question']] = sol_match

all_data = []
with open(processed_path, 'r', encoding='utf-8') as f:
    for line in f:
        all_data.append(json.loads(line))
    
acc = 0
for data in all_data:
    sol_match = ground_truth_dict[data['results']['results']['meta_data']['text']]
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    prediction = data['results']['results']['conversation'][-1]['content'][0]['text']
    cnt = evaluate_metric(ground_truth, prediction)
    print(cnt)
    acc += cnt
    
print(acc)