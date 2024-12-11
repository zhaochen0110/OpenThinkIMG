"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid
import os

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
from functools import partial

from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
from qwen_vl_utils import process_vision_info

from transformers import TextIteratorStreamer, AutoProcessor, Qwen2VLForConditionalGeneration
from threading import Thread

from tool_server.model_workers.base_model_worker import BaseModelWorker

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"qwen2vl_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


class Qwen2ModelWorker(BaseModelWorker):
    
    def init_model(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer

    @torch.inference_mode()
    def generate_stream(self, params):
        conversation = params["conversation"]
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)
        kwargs = {
            **inputs,  # 解包原有的 inputs 字典
            "streamer":streamer,
            'max_new_tokens': min(int(params.get("max_new_tokens", 256)), 1024)  # 添加额外的参数
        }
        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=40001)
    parser.add_argument("--worker-address", type=str,
        default="auto")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:20001")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-name", type=str, default="Qwen2-VL-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = Qwen2ModelWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        model_path=args.model_path,
        model_name=args.model_name,
        device=args.device,
        limit_model_concurrency=args.limit_model_concurrency,
        load_8bit = args.load_8bit, 
        load_4bit = args.load_4bit,
        host = args.host,
        port = args.port,
        no_register = args.no_register,
    )
    worker.run()
