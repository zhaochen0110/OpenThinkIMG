import json
from transformers import AutoTokenizer, set_seed, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = '/mnt/petrelfs/share_data/songmingyang/model/mm/Qwen2.5-VL-3B-Instruct'
data_file = '/mnt/petrelfs/share_data/suzhaochen/chartgemma-reachqa-combined-sharegpt-filterd.json'
# data_file = 'test.json'

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

with open(data_file, 'r', encoding='utf-8') as f:
    all_data = json.load(f)
    
def convert_example(example):
    messages = []
    
    SYSTEM_PROMPT = ("""You are a visual assistant capable of generating and solving steps for chart-based reasoning. Your goal is to answer chart-related questions. You can rely on your own capabilities or use external tools to assist in solving. Here are the available actions:
    - **OCR**: Extracts text from an image. Example: `{"name": "OCR", "arguments": {"image": "img_1"}}`
    - **Point**: Identifies a point in the image based on description and returns coordinates. Example: `{"name": "Point", "arguments": {"image": "img_1", "param": "x-axis value 1970"}}`
    - **ZoomInSubfigure**: Crops the image to the specified subfigure. Example: `{"name": "ZoomInSubfigure", "arguments": {"image": "img_1", "param": "Downstream vs. Concept: Toy"}}`
    - **SegmentRegionAroundPoint**: Segments a region around a given point. Example: `{"name": "SegmentRegionAroundPoint", "arguments": {"image": "img_1", "param": "x=\"21.5\" y=\"28.5\""}}`
    - **DrawHorizontalLineByY**: Draws a horizontal line at a given y-coordinate. Example: `{"name": "DrawHorizontalLineByY", "arguments": {"image": "img_1", "param": "y=28.5"}}`
    - **DrawVerticalLineByX**: Draws a vertical line at a given x-coordinate. Example: `{"name": "DrawVerticalLineByX", "arguments": {"image": "img_1", "param": "x=21.5"}}`
    - **Terminate**: Ends the task and provides the final answer. Example: `{"name": "Terminate", "arguments": {"ans": "1985"}}`

    To solve the problem:
    1. Select actions from the provided tools list, combining them logically and building on previous steps. Call one action at a time, using its output for the next.
    2. To use `SegmentRegionAroundPoint`, `DrawHorizontalLineByY`, or `DrawVerticalLineByX`, first call "Point" to get coordinates for further actions.

    Your output should be in a strict JSON format as follows:
    {"thought": "the reasoning process", "actions": [{"name": "action", "arguments": {"argument1": "value1", "argument2": "value2"}}]}
    """)

    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
    })

    conversations = example.get('conversations')
    image_paths = example.get('images')
    img_idx = 0
    
    for item in conversations:
        content = []
        if item['from'] == 'human':
            role = 'user'
        else:
            role = 'assistant'
        if "You are a visual assistant capable of generating and solving steps" in item['value']:
            content.append({'type':'text', 'text':item['value'].split("\n\nQuestion: ")[-1]})
        else:
            content.append({'type':'text', 'text':item['value']})
        
        if '<image>' in item['value']:
            content.append({'type':'image', 'image':image_paths[img_idx]})
            img_idx += 1
        
        messages.append({
            'role': role,
            'content': content
        })
    
    example["messages"] = messages

    return example

format_data = []
for data in all_data:
    format_data.append(convert_example(data))

with open('count.json', 'a', encoding='utf-8') as f:
    for data in format_data:
        try:
            text_inputs = []
            img_inputs = []
            image_inputs, video_inputs = process_vision_info(data['messages'])
            img_inputs.append(image_inputs)
            text_inputs.append(processor.apply_chat_template(data['messages'], tokenize=False, add_generation_prompt=True))
            batch = processor(
            text=text_inputs,
            images=img_inputs,
            return_tensors="pt",
            padding=True,
            )
            data['text_token'] = batch['input_ids'].shape
            data['pixel_values'] = batch['pixel_values'].shape
            f.write(json.dumps(data)+'\n')
        except Exception as e:
            print(data)
# print(text_inputs)
# print(img_inputs)
# batch = processor(
#     text=text_inputs,
#     images=img_inputs,
#     return_tensors="pt",
#     padding=True,
# )
# print(batch['input_ids'].shape)
# print(batch['pixel_values'].shape)
    

# batch = processor(
#     text=texts,
#     images=image_inputs,
#     return_tensors="pt",
#     padding=True,
# )