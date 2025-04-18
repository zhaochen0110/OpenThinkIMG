import json
import matplotlib.pyplot as plt

# 读取 JSON 文件并存储数据
all_data = []
with open('count.json', 'r', encoding='utf-8') as f:
    for line in f:
        all_data.append(json.loads(line))  # 修正为解析每一行

flag = []
for data in all_data:
    if data['text_token'][1] > 4000 or data['pixel_values'][0] > 20000:
        flag.append(data['conversations'])
        print(data)
        print('***************************************************')
        
with open('chartgemma-reachqa-combined-sharegpt-filterd.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

selected_data = []
for data in train_data:
    if data['conversations'] not in flag:
        selected_data.append(data)
    else:
        print(1)
    
with open('chartgemma-reachqa-combined-sharegpt-filterd-1.json', 'w', encoding='utf-8') as f:
    json.dump(selected_data, f, indent = 4)
            
