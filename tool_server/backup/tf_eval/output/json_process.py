import json

with open('qwenvl_charxiv_10turns.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

new_data = {}
for item in data:
    new_data[item['idx']] = {}
    new_data[item['idx']]['figure_id'] = item['figure_id']
    new_data[item['idx']]['query'] = item['query']
    new_data[item['idx']]['answer'] = item['answer']
    new_data[item['idx']]['final_answer'] = item['value_list'][-1]

with open('qwenvl_charxiv_10turns_final.json', 'w') as f:
    f.write(json.dumps(new_data, indent=2))
