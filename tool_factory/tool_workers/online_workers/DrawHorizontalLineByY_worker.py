"""
A model worker executes the model.
"""

import uuid
import os
import re
import io
import argparse
import torch
import numpy as np
from PIL import Image
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
import matplotlib.pyplot as plt

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None

np.random.seed(3)

# def extract_points(generate_param, image_w, image_h):
#     all_points = []
#     for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', generate_param):
#         try:
#             point = [float(match.group(i)) for i in range(1, 3)]
#         except ValueError:
#             continue
#         else:
#             point = np.array(point)
#             if np.max(point) > 100:
#                 continue
#             point /= 100.0
#             point = point * np.array([image_w, image_h])
#             all_points.append(point)
#     return all_points

def extract_points(generate_param, image_w, image_h):
    all_points = []
    
    # Regular expression to match x and y values separately or together, with or without quotes
    pattern = re.compile(r'(x\d*)?=\s*"?([0-9]+(?:\.[0-9]+)?)"?|'
                         r'(y\d*)?=\s*"?([0-9]+(?:\.[0-9]+)?)"?')
    
    # Initialize default x and y
    points = {}
    for match in pattern.finditer(generate_param):
        attr, x_val, _, y_val = match.groups()
        
        if attr and 'x' in attr:
            points[attr] = float(x_val)
        elif _ and 'y' in _:
            points[_] = float(y_val)
    
    # Process matched pairs
    indices = sorted(set(int(key[1:]) for key in points.keys() if key[1:].isdigit()))
    if not indices:
        indices = [0] if 'x' in points or 'y' in points else []
    
    for i in indices:
        x_key = f'x{i}' if f'x{i}' in points else 'x'
        y_key = f'y{i}' if f'y{i}' in points else 'y'
        
        x_value = points.get(x_key, 0.0)
        y_value = points.get(y_key, 0.0)
        
        point = np.array([x_value, y_value])
        if np.max(point) > 100:
            continue
        point /= 100.0
        point = point * np.array([image_w, image_h])
        all_points.append(point)
    
    return all_points
    
def DrawHorizontalLine(image, point_coords=None):
    image_format = image.format.lower() if image.format else 'png'
    if image_format not in ['png', 'jpeg', 'jpg']:
        image_format = 'png'
        
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    
    for point in point_coords:
        y = point[1]
        if 0 <= y < image.height:
            ax.axhline(y=y, color='#e6194b', linewidth=2, linestyle='dashed')
    
    buf = BytesIO()
    fig.savefig(buf, format=image_format, bbox_inches='tight', pad_inches=0)  
    plt.close(fig) 
    buf.seek(0)
    
    edited_image = Image.open(buf).convert("RGB")
    
    return edited_image

class DrawHorizontalLineToolWorker(BaseToolWorker):
    def __init__(self, 
                 controller_addr, 
                 worker_addr = "auto",
                 worker_id = worker_id, 
                 no_register = False,
                 model_name = "DrawHorizontalLineByY",
                 model_path = "", 
                 model_base = "", 
                 load_8bit = False, 
                 load_4bit = False, 
                 device = "",
                 limit_model_concurrency = 1,
                 host = "0.0.0.0",
                 port = None,
                 model_semaphore = None,
                 ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            model_path,
            model_base,
            model_name,
            load_8bit,
            load_4bit,
            device,
            limit_model_concurrency,
            host,
            port,
            model_semaphore
            )
    def init_model(self):
        logger.info(f"No need to initialize model {self.model_name}.")
        self.model = None

        
    def generate(self, params):
        generate_param = params["param"]
        image = params["image"]
        
        if generate_param is None or image is None:
            logger.error("Missing 'param' or 'image' in the input parameters.")
            return {"text": "Missing 'param' or 'image' in the input parameters.", "edited_image": None}
        
        try:
            img = base64_to_pil(image).convert("RGB")

            width, height = img.size

            ret = {"text": "", "error_code": 0}

            points = extract_points(generate_param, width, height)

            if not points:
                logger.error("No valid points extracted.")
                ret["text"] = "No valid points extracted."
                ret['edited_image'] = None
                return ret
            
            edited_img = DrawHorizontalLine(img, point_coords=points)
            
            ret['text'] = "Line drawn successfully."
            ret['edited_image'] = pil_to_base64(edited_img)
            
        except Exception as e:
            logger.error(f"Error when drawing line: {e}")
            ret["text"] = f"Error when drawing line: {e}"
            ret['edited_image'] = None

        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20037)
    parser.add_argument("--worker-address", type=str,
        default="auto")
    parser.add_argument("--controller-address", type=str,
        default="http://SH-IDCA1404-10-140-54-119:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")


    worker = DrawHorizontalLineToolWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        limit_model_concurrency=args.limit_model_concurrency,
        host = args.host,
        port = args.port,
        no_register = args.no_register
    )
    worker.run()