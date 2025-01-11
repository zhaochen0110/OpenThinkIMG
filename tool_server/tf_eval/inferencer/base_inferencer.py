from gradio.helpers import Examples
import argparse
import base64
from collections import defaultdict
import copy
import datetime
from functools import partial
import json
import os
import torch
from pathlib import Path
import cv2
import numpy as np
import re
import time
from io import BytesIO
from PIL import Image
from PIL import Image as _Image  # using _ to minimize namespace pollution

import gradio as gr
from gradio import processing_utils, utils
from gradio_client import utils as client_utils

import requests

from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
from .dataset import dataset_dict

import pycocotools.mask as mask_util
import uuid

inferencer_id = str(uuid.uuid4())[:6]
logger = build_logger("base_inferencer", f"base_inferencer_{inferencer_id}.log")

R = partial(round, ndigits=2)

class BaseInferencer():
    def __init__(
        self,
        inference_args,
        data_args,
    ):
        self.inference_args = inference_args
        self.data_args = data_args
        
        self.controller_addr = inference_args.controller_addr
        self.available_models = self.get_model_list()
        self.headers = {"User-Agent": "LLaVA-Plus Client"}
        self.init_model_addr_dict()
    
    
    def get_model_list(self):
        ret = requests.post(self.controller_addr + "/refresh_all_workers")
        assert ret.status_code == 200
        ret = requests.post(self.controller_addr + "/list_models")
        models = ret.json()["models"]
        logger.info(f"Models: {models}")
        return models
    
    def get_worker_addr(self, model_name):
        return self.model_addr_dict[model_name]
        
        
    def init_model_addr_dict(self):
        self.model_addr_dict = {}
        for model_name in self.available_models:
            ret = requests.post(self.controller_addr + "/get_worker_address",
                                json={"model": model_name})
            worker_addr = ret.json()["address"]
            if worker_addr == "":
                logger.error(f"worker_addr for {model_name} is empty")
                continue
            self.model_addr_dict[model_name] = worker_addr
    
    def refresh_workers(self):
        self.available_models = self.get_model_list()
        self.init_model_addr_dict()
    
    def model_specific_process_conversation(self, 
                                            text_prompt, 
                                            image, 
                                            role) -> object:
        pass
    
    
    def model_specific_append_message_to_conversation(self, 
                                                      conversation, 
                                                      text_prompt, 
                                                      image, 
                                                      role) -> object:
        pass
    
    
    def get_model_response(self, model_name, conversation, image=None, gen_kwargs={}):
        pass
    
    
    def parse_tool_config(self, output,):
        ### Parse Tool Config
        try:
            pattern = r'"thoughts🤔"(.*)"actions🚀"(.*)"value👉"(.*)'
            matches = re.findall(pattern, output, re.DOTALL)
            # import ipdb; ipdb.set_trace()
            if len(matches) > 0:
                # tool_cfg = json.loads(matches[0][1].strip())
                try:
                    tool_cfg = json.loads(matches[0][1].strip())
                except Exception as e:
                    tool_cfg = json.loads(
                        matches[0][1].strip().replace("\'", "\""))
                print("tool_cfg:", tool_cfg)
            else:
                tool_cfg = None
        except Exception as e:
            logger.info(f"Failed to parse tool config: {e}")
            tool_cfg = None
            
        print("trigger tool augmentation with tool_cfg: ", tool_cfg)
        return tool_cfg 
    
    
    def get_tool_response(self, tool_cfg, image):

        if tool_cfg is not None and len(tool_cfg) > 0:
            try:
                assert len(tool_cfg) == 1, "Only one tool is supported for now, but got: {}".format(tool_cfg)
                api_name = tool_cfg[0].get("API_name", tool_cfg[0].get("api_name", ""))
                if api_name not in self.available_models:
                    logger.error(f"API_name {api_name} not in available models, {self.available_models}")
                    return dict(text=f"There is no tool names {api_name}.")
                if api_name in ["line","ocr","crop","grounding","grounding_dino"]:
                    assert image is not None, "Image is required for tool: {}".format(api_name)
                    image = load_image(image)
                    image = pil_to_base64(image)
                else:
                    image = None
                tool_cfg[0].get("api_params",tool_cfg[0].get("API_params",{})).pop('image', None)
                api_params = tool_cfg[0].get("api_params",tool_cfg[0].get("API_params",{}))
                # image is a base64 img obj
                api_paras = {
                    "box_threshold": 0.3,
                    "text_threshold": 0.25,
                    **api_params,
                }
                
                if image:
                    api_paras['image'] = image
                
                tool_worker_addr = self.get_worker_addr(api_name)
                print("tool_worker_addr: ", tool_worker_addr)
                tool_response = requests.post(
                    tool_worker_addr + "/worker_generate",
                    headers=self.headers,
                    json=api_paras,
                ).json()
                tool_response_clone = copy.deepcopy(tool_response)
                if "edited_image" in tool_response:
                    tool_response.pop("edited_image", None)
                print("tool_response: ", tool_response)
                
                return tool_response_clone
            except:
                logger.error(f"Tool {api_name} failed to answer the question.")
                return dict(text=f"Tool {api_name} failed to answer the question.")
        else:
            logger.error(f"Tool {api_name} failed to answer the question.")
            return dict(text=f"Tool {api_name} failed to answer the question.")

    
    
    def inference_on_one_instance(
        self,
        instance,
        model_name,
        max_rounds=3
    ): 
        prompt = instance["prompt"]
        image = instance["image"]
        gen_kwargs = instance["gen_kwargs"]
        
        ### Initialize log viriables
        current_round = 0
        conversation_logs = []
        generation_logs = []
        
        pil_image = load_image(image)
        base64_image = pil_to_base64(pil_image)
        
        
        original_prompt = copy.deepcopy(prompt)
        # image here can be both img_path or base64
        conversation = self.model_specific_process_conversation(
            text_prompt=prompt, image=image, role="user"
        )
        conversation_logs.append(conversation)
        
        ## First Round Model Responding
        lm_output = self.get_model_response(model_name, conversation, image, gen_kwargs)
        generation_logs.append(lm_output)
        
        tool_cfg = self.parse_tool_config(lm_output)
        
        ## While Tool Config is not None, keep triggering tool augmentation
        while lm_output is not None and "<stop>" not in lm_output and current_round < max_rounds:
            current_round += 1
            conversation = self.model_specific_append_message_to_conversation(
                conversation=conversation, text_prompt=lm_output, image=None, role="assistant"
            )
            
            
            if tool_cfg:
                try:
                    tool_response = self.get_tool_response(tool_cfg, image)

                    ## 将图片提取出来
                    if "edited_image" in tool_response:
                        edited_image = tool_response.pop("edited_image")
                        edited_image = "data:image/jpeg;base64," + edited_image
                    else:
                        edited_image = None
                    
                    if "text" in tool_response:
                        tool_response_text = tool_response["text"]
                    else:
                        tool_response_text = None
                        
                    api_name = tool_cfg[0].get("API_name", tool_cfg[0].get("api_name", ""))
                    new_response = f"{api_name} model outputs: {tool_response_text}\n\n"
                    new_round_conv = f"{new_response} Please summarize the model outputs and answer my first question: {original_prompt}"
                except:
                    edited_image = None
                    new_round_conv = original_prompt
            else:
                edited_image = None
                new_round_conv = original_prompt
            
            
            # FIXME: edited_image may be empty, this situation should be taken into account
            conversation = self.model_specific_append_message_to_conversation(
                conversation=conversation, text_prompt=new_round_conv, image=edited_image, role="user"
            )

            conversation_logs.append(conversation)
            # print(conversation)
            lm_output = self.get_model_response(model_name, conversation, edited_image, gen_kwargs)
            generation_logs.append(lm_output)
            tool_cfg = self.parse_tool_config(lm_output)
            
        return generation_logs, conversation_logs
    
    
    def single_loop_inference(self):
        assert self.data_args is not None, "data_args is required for single loop inference"
        self.dataset = dataset_dict[self.data_args.dataset_name](self.data_args)
        
        for idx,item in enumerate(tqdm(self.dataset)):
            copy_item = copy.deepcopy(item)
            generation_logs, conversation_logs = self.inference_on_one_instance(
                instance=copy_item, model_name=self.inference_args.model_name, max_rounds=self.inference_args.max_rounds
            )
            
            value_list = [self.parse_lm_response(i)[2] for i in generation_logs if self.parse_lm_response(i)[2] is not None]
            tool_list = [self.parse_lm_response(i)[1]  for i in generation_logs if self.parse_lm_response(i)[1] is not None]
            response_list = generation_logs
            res = dict(value_list=value_list, response_list=response_list, tool_list=tool_list, **item)
            self.dataset.write_output_item(res)
            

    def parse_lm_response(self, response):
        pattern = r'"thoughts🤔"(.*)"actions🚀"(.*)"value👉"(.*)'
        matches = re.findall(pattern, response, re.DOTALL)
        if len(matches) > 0:
            try:
                thoughts = matches[0][0].strip()
            except:
                thoughts = None
            try:
                actions = matches[0][1].strip()
            except:
                actions = None
            try:
                value = matches[0][2].strip()
            except:
                value = None
        else:
            thoughts, actions, value = None, None, None
        return thoughts, actions, value