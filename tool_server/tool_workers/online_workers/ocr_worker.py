"""
A model worker executes the model.
"""

import uuid
import os
import re
import io
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

import easyocr

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None

np.random.seed(3)

class OCRToolWorker(BaseToolWorker):
    def __init__(self, 
                 controller_addr, 
                 worker_addr = "auto",
                 worker_id = worker_id, 
                 no_register = False,
                 model_name = "OCR",
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
        logger.info(f"Initializing model {self.model_name}...")
        self.ocr_model = easyocr.Reader(['ch_sim','en'])

        
    def generate(self, params):
        image = params["image"]
        
        if  image is None:
            logger.error("Missing 'image' for the ocr input parameters.")
            return {"text": "Missing 'image' for the ocr input parameters."}
        
        ret = {"text": "", "error_code": 0}
        
        try:
            img = base64_to_pil(image).convert("RGB")

            result = self.ocr_model.readtext(np.array(img))

            texts = [i[1] for i in result]
            
            ret["text"] = str(texts)
            
        except Exception as e:
            logger.error(f"Error when ocr: {e}")
            ret["text"] = f"Error when ocr: {e}"

        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20009)
    parser.add_argument("--worker-address", type=str,
        default="auto")
    parser.add_argument("--controller-address", type=str,
        default="http://SH-IDCA1404-10-140-54-119:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")


    worker = OCRToolWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        limit_model_concurrency=args.limit_model_concurrency,
        host = args.host,
        port = args.port,
        no_register = args.no_register
    )
    worker.run()