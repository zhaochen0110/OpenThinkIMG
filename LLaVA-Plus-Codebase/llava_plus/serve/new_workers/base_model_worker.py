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


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"base_model_worker_{worker_id}.log")
model_semaphore = None

class BaseModelWorker:
    def __init__(self, 
                 controller_addr, 
                 worker_addr = "auto",
                 worker_id = worker_id, 
                 no_register = False,
                 model_path = None, 
                 model_base = None, 
                 model_name = None,
                 load_8bit = False, 
                 load_4bit = False, 
                 device = "auto",
                 limit_model_concurrency = 1,
                 host = "0.0.0.0",
                 port = None,
                 model_semaphore = None,
                 ):
        self.controller_addr = controller_addr
        assert port is not None, "Port must be specified"
        if worker_addr == "auto":
            node_name = os.getenv("SLURMD_NODENAME", "Unknown")
            print(f"SLURM Node Name: {node_name}")
            assert node_name != "Unknown"
            self.worker_addr = f"http://{node_name}:{port}"
        else:
            self.worker_addr = worker_addr
            
        self.model_path = model_path
        self.model_base = model_base
        if model_name is None:
            self.model_name = model_path.split("/")[-1]
        else:
            self.model_name = model_name
        
        if model_semaphore is not None:
            self.model_semaphore = model_semaphore
        else:
            self.model_semaphore = None
        
        self.worker_id = worker_id
        self.no_register = no_register
        
       
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit
        self.device = device
        self.limit_model_concurrency = limit_model_concurrency
        self.host = host
        self.port = port
        
        self.global_counter = 0
        
        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=self.heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()
            
        # Set up the routes
        self.app = FastAPI()
        self.setup_routes()
        self.init_model()
        
        
    
    def init_model(self):
        pass
    
    def heart_beat_worker(self, controller):

        while True:
            time.sleep(WORKER_HEART_BEAT_INTERVAL)
            controller.send_heart_beat()

        
    def release_model_semaphore(self, fn=None):
        if self.model_semaphore:
            self.model_semaphore.release()
            if fn is not None:
                fn()
                
    def setup_routes(self):
        @self.app.post("/worker_generate_stream")
        async def generate_stream(request: Request):
            self.global_counter += 1
            params = await request.json()

            if self.model_semaphore is None:
                self.model_semaphore = asyncio.Semaphore(self.limit_model_concurrency)
            await self.model_semaphore.acquire()
            
            self.send_heart_beat()
            generator = self.generate_stream_gate(params)
            background_tasks = BackgroundTasks()
            background_tasks.add_task(
                partial(self.release_model_semaphore, fn=self.send_heart_beat)
            )
            return StreamingResponse(generator, background=background_tasks)

        @self.app.post("/worker_get_status")
        async def get_status(request: Request):
            return self.get_status()
        
    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200, f"Failed to register to controller: {r.text}"

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {self.global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return self.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        pass
    
    
    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
    
    
    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")