"""
A model worker executes the model.
"""

import uuid
import os
import re
import io
import argparse
import torch

from PIL import Image, ImageDraw
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
import matplotlib.pyplot as plt

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

from tool_server.utils.cogcom.models.cogcom_model import CogCoMModel
from tool_server.utils.cogcom.utils import chat
from tool_server.utils.cogcom.utils import get_image_processor, llama2_tokenizer, llama2_text_processor_inference

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None

class CogComGroundingToolWorker(BaseToolWorker):
    def __init__(self, 
                 controller_addr, 
                 worker_addr = "auto",
                 worker_id = worker_id, 
                 no_register = False,
                 model_path = "/mnt/petrelfs/share_data/quxiaoye/models/cogcom-grounding-17b", 
                 model_base = "/mnt/petrelfs/share_data/quxiaoye/models/vicuna-7b-v1.5", 
                 model_name = "grounding",
                 load_8bit = False, 
                 load_4bit = False, 
                 device = "auto",
                 limit_model_concurrency = 1,
                 host = "0.0.0.0",
                 port = None,
                 model_semaphore = None,
                 max_length = 2048,
                 top_p = 0.4,
                 top_k = 1,
                 english = False, # only output English
                 version = 'cogcom-base', # version of the model you want to load
                 fp16 = False,
                 bf16 = False,
                 add_preprompt = False,
                 parse_result = False,
                 temperature = 0.1,
                 no_prompt = False, # Sometimes there is no prompt in stage 1
                 ):
        self.max_length = max_length
        self.top_p = top_p
        self.top_k = top_k
        self.version = version
        self.fp16 = fp16
        self.bf16 = bf16
        self.add_preprompt = add_preprompt
        self.parse_result = parse_result
        self.temperature = temperature
        self.no_prompt = no_prompt
        self.english = english
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
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        model, model_args = CogCoMModel.from_pretrained(
            self.model_path,
            args=argparse.Namespace(
                deepspeed=None,
                local_rank=self.rank,
                rank=self.rank,
                world_size=self.world_size,
                model_parallel_size=self.world_size,
                mode='inference',
                skip_init=True,
                use_gpu_initialization=True if torch.cuda.is_available() else False,
                device='cuda',
                bf16=False,
                fp16=False,
            ), url='local' if os.path.exists(self.model_path) else None,
            overwrite_args={'model_parallel_size': self.world_size} if self.world_size != 1 else {})

        self.model = model.eval()
        from sat.mpu import get_model_parallel_world_size
        assert self.world_size == get_model_parallel_world_size(), "World size must equal to model parallel size for cli_demo!"

        self.tokenizer = llama2_tokenizer(self.model_base, signal_type="chat")
        self.image_processor = get_image_processor(490)
        self.cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None
        self.text_processor_infer = llama2_text_processor_inference(
            self.tokenizer, 
            self.max_length, 
            self.model.image_length if hasattr(self.model, 'image_length') else 0, 
            self.model, 
            self.no_prompt, 
            self.english
        )

        
    def generate(self, params):
        generate_param = params["param"]
        image = params["image"]
        image =  base64_to_pil(image) #  PIL image

        text_prompt = "Find the region in image that {} describes.".format(generate_param)
        
        ret = {"text": "", "error_code": 0}
        try:
            with torch.no_grad():
                history = None
                cache_image = None
                response, history, cache_image = chat(
                    image_path = "",
                    model=self.model,
                    text_processor=self.text_processor_infer,
                    img_processor=self.image_processor,
                    cross_img_processor=self.cross_image_processor,
                    query=text_prompt,
                    image=image,
                    history=history,
                    max_length=self.max_length,
                    top_p=0.9,  # Default nucleus sampling
                    temperature=self.temperature,
                    top_k=50,  # Default top-k sampling
                    invalid_slices=self.text_processor_infer.invalid_slices,
                    no_prompt=self.no_prompt,
                    add_preprompt=False,  # Default: no preprompt
                    parse_result=True
                )
                if response is not None:
                    ret["text"] = response
                    pattern = r"\[\[(\d+),(\d+),(\d+),(\d+)\]\]"
                    matches = re.findall(pattern, response)
                    if matches:
                        draw = ImageDraw.Draw(image)
                        width, height = image.size

                        for match in matches:
                            box = [int(num) for num in match]

                            x1 = box[0] / 1000 * width
                            y1 = box[1] / 1000 * height
                            x2 = box[2] / 1000 * width
                            y2 = box[3] / 1000 * height
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                            
                        ret['edited_image'] = pil_to_base64(image)
                    else:
                        ret["text"] = "No region found in image."
                        ret['edited_image'] = None
                else:
                    ret["text"] = "No region found in image."
                    ret['edited_image'] = None

        except Exception as e:
            logger.error(f"Error when using cogcom to ground: {e}")
            ret["text"] = f"Error when using cogcom to ground: {e}"
            ret['edited_image'] = None
        
        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20017)
    parser.add_argument("--worker-address", type=str,
        default="auto")
    parser.add_argument("--controller-address", type=str,
        default="http://SH-IDCA1404-10-140-54-89:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--model_path", type=str, default="/mnt/petrelfs/share_data/quxiaoye/models/cogcom-grounding-17b")
    parser.add_argument("--model_base", type=str, default="/mnt/petrelfs/share_data/quxiaoye/models/vicuna-7b-v1.5")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = CogComGroundingToolWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        limit_model_concurrency=args.limit_model_concurrency,
        host = args.host,
        port = args.port,
        no_register = args.no_register,
        model_path = args.model_path,
        model_base = args.model_base
    )
    worker.run()