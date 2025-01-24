import json
import os
import base64

# 输入文件路径
file_path = "/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/gemini_caogao.jsonl"

# 输出文件夹
output_base_dir = "./processed_data"
output_json_path = os.path.join(output_base_dir, "all_results.json")

# 确保输出目录存在
os.makedirs(output_base_dir, exist_ok=True)

# 存储所有处理后的数据
all_processed_data = []

# 读取 JSONL 文件
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)

        if "results" in data:
            results = data["results"]
            for item in results:
                if 'idx' in item:
                    data_id = item['idx']  # 使用 idx 作为唯一ID
                else:
                    continue  # 如果没有 idx，跳过当前条目

                image_output_dir = os.path.join(output_base_dir, "images", str(data_id))
                os.makedirs(image_output_dir, exist_ok=True)

                processed_conversation = []
                conversation = item.get('results', {}).get('conversation', [])
                image_counter = 1  # 图片编号计数

                for msg in conversation:
                    new_msg = msg.copy()
                    if 'content' in msg:
                        new_content_list = []
                        for content in msg['content']:
                            new_content = content.copy()

                            if content['type'] == 'image_url' and 'image_url' in content:
                                image_data = content['image_url']['url']
                                if image_data.startswith("data:image"):
                                    header, encoded = image_data.split(",", 1)
                                    file_extension = header.split('/')[1].split(';')[0]
                                    image_filename = f"{image_counter}.{file_extension}"
                                    image_path = os.path.join(image_output_dir, image_filename)
                                    
                                    with open(image_path, "wb") as img_file:
                                        img_file.write(base64.b64decode(encoded))
                                    
                                    new_content['image_url']['url'] = image_path
                                    image_counter += 1

                            elif content['type'] == 'text' and '[BEGIN OF GOAL]' in content['text']:
                                new_content['text'] = content['text'].split("Question: ")[-1]

                            new_content_list.append(new_content)

                        new_msg['content'] = new_content_list

                    processed_conversation.append(new_msg)

                all_processed_data.append({
                    "id": data_id,
                    "conversation": processed_conversation
                })

# 将所有数据写入到一个 JSON 文件
with open(output_json_path, 'w', encoding='utf-8') as json_out:
    json.dump(all_processed_data, json_out, ensure_ascii=False, indent=4)

print(f"处理完成，所有数据保存在: {output_json_path}")
print(f"图片按ID存储在: {os.path.join(output_base_dir, 'images')}")
