import json
import random
import cv2
import numpy as np
from PIL import Image
import uuid
import argparse
import requests
import time
import re

from tool_server.utils.utils import *
from tool_server.utils.server_utils import *

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None

def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def get_sorted_bbox(img):
    image = np.array(img)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    gaussian = cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)
    edged = cv2.Canny(gaussian, 100, 200) 
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key= len, reverse=True)[:10]
    longest_contour = sorted_contours[0]
    x,y,w,h = cv2.boundingRect(np.concatenate(longest_contour))
    longest_box = w + h

    sorted_bbox = []
    for c in sorted_contours:
        x,y,w,h = cv2.boundingRect(np.concatenate(c))
        if longest_box - 10 < w + h <= longest_box + 10:
            repeat = False
            for bb in sorted_bbox:
                if abs(bb['x1'] - x) < 10 and abs(bb['y1'] - y) < 10 and abs(bb['x2'] - (x+w)) < 10 and abs(bb['y2'] - (y+h)) < 10:
                    repeat = True
            if not repeat:
                sorted_bbox.append({'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h})
    sorted_bbox_json = json.dumps(sorted_bbox)
    
    return sorted_bbox_json

def get_subplot_images(img, sorted_bbox_json):
    sorted_bbox = json.loads(sorted_bbox_json)

    image = np.array(img)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    result_images = []

    for bbox in sorted_bbox:
        
        masked_image = image.copy()
        
        for other_bbox in sorted_bbox:
            if other_bbox != bbox:
                x1, y1, x2, y2 = other_bbox['x1'], other_bbox['y1'], other_bbox['x2'], other_bbox['y2']
                cv2.rectangle(masked_image, (x1, y1), (x2, y2), (255, 255, 255), -1)

        pil_image = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        result_images.append(pil_image)

    return result_images
    
def create_gemini_json(param, img_list):
    text_content = f"""
    An image chart may contain several subplots. To facilitate the observation of chart information, the image has been processed into multiple images, each containing only one subplot.  
    Find out which image is most related to {param}.  
    You are only allowed to output text like "image-i" and do not need to output any other reasoning processes.
    """

    contents = [{"parts": []}]

    for idx, image in enumerate(img_list):
        base64_data = encode_image_to_base64(image)
        contents[0]["parts"].extend(
            [
                {
                    "text": f"image-{idx+1}:\n"
                },
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64_data
                    }
                }
            ]
        )

    contents[0]["parts"].extend(
        [
            {
                "text": text_content
            }
        ]
    )

    data = {
        "contents": contents
    }
    
    return data

def call_gemini(data):
    api_key = "AIzaSyDlmb73omTgAGvw_a9lGxK5pC56fuLtJoQ"  # 替换为你的 API 密钥

    proxies = {
        "http": "http://closeai-proxy.pjlab.org.cn:23128",  # 代理地址
        "https": "http://closeai-proxy.pjlab.org.cn:23128",  # 代理地址
    }
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json",
    }
    
    attempt = 0
    while attempt < 5:
        try:
            response = requests.post(url, headers=headers, json=data, proxies=proxies)

            if response.status_code != 200:
                raise Exception(f"Error: {response.status_code}, {response.text}")
            
            response = response.json()
            print("text response: ", response["candidates"][0]["content"]["parts"][0]["text"])
            
            return response["candidates"][0]["content"]["parts"][0]["text"]

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}, sleep 30s and try again!")
            sleep_time = 30
            time.sleep(sleep_time)
            attempt += 1
    
    return "Some thing wrong and there is nothing return"

def extract_image_ids(response):
    try:
        matches = [int(num) for num in re.findall(r'image-(\d+)', response)]
        return matches
    
    except Exception as e:
        logging.error(f"An error occurred when extract image_ids in gemini outputs: {e}")
        return []

    
class SelectSubplotToolWorker(BaseToolWorker):
    def __init__(self, 
                 controller_addr, 
                 worker_addr = "auto",
                 worker_id = worker_id, 
                 no_register = False,
                 model_name = "SelectSubplot",
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
            return {"text": "Missing 'param' or 'image' in the input parameters.", "subplot_images": None}
        
        ret = {"text": "", "error_code": 0}
        
        try:
            img = base64_to_pil(image).convert("RGB")
            
            # get top-10 contours
            sorted_bbox_json = get_sorted_bbox(img=img)
            
            # get subplot-images
            subplots_list = get_subplot_images(img, sorted_bbox_json)
            
            # creat call-gemini prompts
            data = create_gemini_json(generate_param, subplots_list)
            
            # call gemini
            response = call_gemini(data)
            
            # extract image ids from outputs
            img_ids = extract_image_ids(response)
            
            # 
            img_list = []
            try:
                for num in img_ids:
                    if num - 1 < 0 or num - 1 >= len(subplots_list):
                        raise IndexError(f"Index out of range: {num - 1}")
                    base64_img = encode_image_to_base64(subplots_list[num - 1])
                    img_list.append(base64_img)
            except IndexError as e:
                logging.error(f"An error occurred: {e}")
            
            ret["text"] = "Select subplot done."
            ret['subplot_images'] = img_list
            
        except Exception as e:
            logger.error(f"Error when selecting subplot: {e}")
            ret["text"] = f"Error when selecting subplot: {e}"
            ret['subplot_images'] = None

        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20038)
    parser.add_argument("--worker-address", type=str,
        default="auto")
    parser.add_argument("--controller-address", type=str,
        default="http://SH-IDCA1404-10-140-54-119:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = SelectSubplotToolWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        limit_model_concurrency=args.limit_model_concurrency,
        host = args.host,
        port = args.port,
        no_register = args.no_register
    )
    worker.run()
    
    
    

    