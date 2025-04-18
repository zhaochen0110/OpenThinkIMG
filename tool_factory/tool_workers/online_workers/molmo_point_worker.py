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

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None

def extract_points(molmo_output, image_w, image_h):
    all_points = []
    for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', molmo_output):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    return np.array(all_points)  # Ensure it's always a NumPy array

def show_points(coords, labels, ax, marker_size=375):
    # Only plot if there are points
    if len(coords) == 0:
        return
    
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def create_image_with_points(image, coords, labels, marker_size=375):
    fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100), dpi=100)
    ax.imshow(image)
    image_format = image.format.lower()
    if image_format not in ['png', 'jpeg', 'jpg']:
        image_format = 'png'

    # Only show points if there are any valid coordinates
    show_points(coords, labels, ax, marker_size)

    plt.axis('off')  # Turn off axis

    # Convert the figure to a PIL image
    buf = BytesIO()
    plt.savefig(buf, format=image_format, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

class MolmoToolWorker(BaseToolWorker):
    def __init__(self, 
                 controller_addr, 
                 worker_addr = "auto",
                 worker_id = worker_id, 
                 no_register = False,
                 model_path = "/mnt/petrelfs/share_data/suzhaochen/models/Molmo-7B-D-0924", 
                 model_base = "", 
                 model_name = "Point",
                 load_8bit = False, 
                 load_4bit = False, 
                 device = "auto",
                 limit_model_concurrency = 1,
                 host = "0.0.0.0",
                 port = None,
                 model_semaphore = None,
                 max_length = 2048,
                 ):
        self.max_length = max_length
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
        logger.info(f"Initializing model {self.model_name}...")

        # load the processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        
    def generate(self, params):
        generate_param = params["param"]
        image = params["image"]
        image =  base64_to_pil(image) #  PIL image
        # breakpoint()
        text_prompt = "Point to the {} in the scene.".format(generate_param)
        
        ret = {"text": "", "error_code": 0}
        try:
            with torch.no_grad():
                inputs = self.processor.process(
                    images=[image],
                    text=text_prompt,
                )
                inputs["images"] = inputs["images"].to(torch.bfloat16)

                inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):

                        output = self.model.generate_from_batch(
                            inputs,
                            GenerationConfig(max_new_tokens=self.max_length, stop_strings="<|endoftext|>"),
                            tokenizer=self.processor.tokenizer
                        )

                        # only get generated tokens; decode them to text
                        generated_tokens = output[0,inputs['input_ids'].size(1):]
                        response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        
                        if response is not None:
                            ret["text"] = response
                        #     width, height = image.size
                        #     all_points = extract_points(response, width, height)
                            
                        #     if all_points.size > 0:
                        #         input_labels = np.ones(len(all_points))
                        #         image = create_image_with_points(image, all_points, input_labels)
                        #         ret['edited_image'] = pil_to_base64(image)
                        #     else:
                        #         ret["text"] = "No region found in image."
                        #         ret['edited_image'] = None
                        # else:
                        #     ret["text"] = "No region found in image."
                        #     ret['edited_image'] = None
                
        except Exception as e:
            logger.error(f"Error when using molmo to point: {e}")
            ret["text"] = f"Error when using molmo to point: {e}"
            # ret['edited_image'] = None
        
        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20027)
    parser.add_argument("--worker-address", type=str,
        default="auto")
    parser.add_argument("--controller-address", type=str,
        default="http://SH-IDCA1404-10-140-54-119:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--model_path", type=str, default="/mnt/petrelfs/share_data/suzhaochen/models/Molmo-7B-D-0924")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = MolmoToolWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        limit_model_concurrency=args.limit_model_concurrency,
        host = args.host,
        port = args.port,
        no_register = args.no_register,
        model_path = args.model_path,
    )
    worker.run()