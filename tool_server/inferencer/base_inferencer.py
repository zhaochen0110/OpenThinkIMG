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


import pycocotools.mask as mask_util
import uuid

inferencer_id = str(uuid.uuid4())[:6]
logger = build_logger("base_inferencer", f"base_inferencer_{inferencer_id}.log")

R = partial(round, ndigits=2)

class BaseInferencer():
    def __init__(
        self,
        controller_addr = "http://localhost:20001",
    ):
        self.controller_addr = controller_addr
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
    
    def model_specific_process_conversation(self, message, images, role):
        pass
    
    
    def model_specific_append_message_to_conversation(self, message, images, role):
        pass
    
    
    def get_model_response(self, model_name, conversation, image, gen_kwargs):
        assert model_name in self.available_models
        assert isinstance(gen_kwargs, dict)
        
        if isinstance(image, str):
            image = Image.open(image)
        assert isinstance(image, Image.Image)
        image_base64 = pil_to_base64(image)
        pload = {
            "model": model_name,
            "prompt": conversation,
            "temperature": gen_kwargs.get("temperature", 0.0),
            "top_p": gen_kwargs.get("top_p", 1.0),
            "max_new_tokens": gen_kwargs.get("max_new_tokens", 2048),
            "images": image_base64,
        }
        
        logger.info(f"==== request ====\n{pload}\n==== request ====")
        try:
            worker_addr = self.model_addr_dict[model_name]
            response = requests.post(worker_addr + "/worker_generate_stream",
                                    headers=self.headers, json=pload, stream=True, timeout=10)
            # import ipdb; ipdb.set_trace()
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data["error_code"] == 0:
                        output = data["text"].strip()
                        yield output
                    else:
                        output = data["text"] + \
                            f" (error_code: {data['error_code']})"
                        yield output
                        return
                    time.sleep(0.03)
        except requests.exceptions.RequestException as e:
            logger.error(f"error: {e}")
            yield f"error: {e}"
            return
        logger.info(f"==== response ====")
        logger.info(output)
        return output
    
    def parse_tool_config(self, output):
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
        if isinstance(image, str):
            image = Image.open(image)
        assert isinstance(image, Image.Image)
        image_base64 = pil_to_base64(image)
        
        
        if tool_cfg is not None and len(tool_cfg) > 0:
            assert len(tool_cfg) == 1, "Only one tool is supported for now, but got: {}".format(tool_cfg)
            api_name = tool_cfg[0]['API_name']
            tool_cfg[0]['API_params'].pop('image', None)
            
            # image is a pil obj
            api_paras = {
                'image': image,
                "box_threshold": 0.3,
                "text_threshold": 0.25,
                **tool_cfg[0]['API_params']
            }
            
            ## DO TOOL ADJUSTMENT
            # if api_name in ['inpainting']:
            #     api_paras['mask'] = getattr(state, 'mask_rle', None)
            # if api_name in ['openseed', 'controlnet']:
            #     if api_name == 'controlnet':
            #         api_paras['mask'] = getattr(state, 'image_seg', None)
            #     api_paras['mode'] = api_name
            #     api_name = 'controlnet'
            # if api_name == 'seem':
            #     reference_image = getattr(state, 'reference_image', None)
            #     reference_mask = getattr(state, 'reference_mask', None)
            #     api_paras['refimg'] = reference_image
            #     api_paras['refmask'] = reference_mask
                # extract ref image and mask
                

            # import ipdb; ipdb.set_trace()
            # breakpoint()
            tool_worker_addr = self.get_worker_addr(self.controller_addr, api_name)
            print("tool_worker_addr: ", tool_worker_addr)
            tool_response = requests.post(
                tool_worker_addr + "/worker_generate",
                headers=self.headers,
                json=api_paras,
            ).json()
            tool_response_clone = copy.deepcopy(tool_response)
            print("tool_response: ", tool_response)
            
            
            # clean up the response
            masks_rle = None
            edited_image = None
            image_seg = None  # for openseed
            iou_sort_masks = None
            if 'boxes' in tool_response:
                tool_response['boxes'] = [[R(_b) for _b in bb]
                                        for bb in tool_response['boxes']]
            if 'logits' in tool_response:
                tool_response['logits'] = [R(_l) for _l in tool_response['logits']]
            if 'scores' in tool_response:
                tool_response['scores'] = [R(_s) for _s in tool_response['scores']]
            if "masks_rle" in tool_response:
                masks_rle = tool_response.pop("masks_rle")
            if "edited_image" in tool_response:
                edited_image = tool_response.pop("edited_image")
            if "size" in tool_response:
                _ = tool_response.pop("size")
            if api_name == "easyocr":
                _ = tool_response.pop("boxes")
                _ = tool_response.pop("scores")
            if "retrieval_results" in tool_response:
                tool_response['retrieval_results'] = [
                    {'caption': i['caption'], 'similarity': R(i['similarity'])}
                    for i in tool_response['retrieval_results']
                ]
            if "image_seg" in tool_response:
                image_seg = tool_response.pop("image_seg")
            if "iou_sort_masks" in tool_response:
                iou_sort_masks = tool_response.pop("iou_sort_masks")
            if len(tool_response) == 0:
                tool_response['message'] = f"The {api_name} has processed the image."
            # hack
            # if masks_rle is not None:
            #     state.mask_rle = masks_rle[0]
            # if image_seg is not None:
            #     state.image_seg = image_seg
            return tool_response_clone

    
    
    def inference_on_one_instance(self,instance,model_name):   
        prompt = instance["prompt"]
        image = instance["image"]
        gen_kwargs = instance["gen_kwargs"]
        original_prompt = copy.deepcopy(prompt)
        conversation = self.model_specific_process_conversation(prompt, image)
        ## First Round Model Responding
        lm_output = self.get_model_response(model_name, conversation, image, gen_kwargs)
        tool_cfg = self.parse_tool_config(lm_output)
        
        ## While Tool Config is not None, keep triggering tool augmentation
        while tool_cfg is not None and len(tool_cfg) > 0:
            tool_response = self.get_tool_response(tool_cfg, image)
            
            ## 将图片提取出来
            if "edited_image" in tool_response:
                edited_image = tool_response.pop("edited_image")
                
            api_name = tool_cfg[0]['API_name']
            new_response = f"{api_name} model outputs: {tool_response}\n\n"
            new_round_conv = new_response + "Please summarize the model outputs and answer my first question: {}".format(original_prompt)
            
            
            conversation = self.model_specific_append_message_to_conversation(new_round_conv, edited_image, "user")
            print(conversation)
            lm_output = self.get_model_response(model_name, conversation, edited_image, gen_kwargs)
            tool_cfg = self.parse_tool_config(lm_output)
            

        #     # remove the cursor
        #     state.messages[-1][-1] = state.messages[-1][-1][:-1]

        #     # add image(s)
        #     if edited_image is not None:
        #         edited_image_pil = Image.open(
        #             BytesIO(base64.b64decode(edited_image))).convert("RGB")
        #         state.messages[-1][-1] = (state.messages[-1]
        #                                 [-1], edited_image_pil, "Crop")
        #     if image_seg is not None:
        #         edited_image_pil = Image.open(
        #             BytesIO(base64.b64decode(image_seg))).convert("RGB")
        #         state.messages[-1][-1] = (state.messages[-1]
        #                                 [-1], edited_image_pil, "Crop")
        #     if iou_sort_masks is not None:
        #         assert isinstance(
        #             iou_sort_masks, list), "iou_sort_masks should be a list, but got: {}".format(iou_sort_masks)
        #         edited_image_pil_list = [Image.open(
        #             BytesIO(base64.b64decode(i))).convert("RGB") for i in iou_sort_masks]
        #         state.messages[-1][-1] = (state.messages[-1]
        #                                 [-1], edited_image_pil_list, "Crop")
        #     if api_name in ['grounding_dino', 'ram+grounding_dino', 'blip2+grounding_dino']:
        #         edited_image_pil = Image.open(
        #             BytesIO(base64.b64decode(state.get_images()[0]))).convert("RGB")
        #         edited_image_pil = plot_boxes(edited_image_pil, tool_response)
        #         state.messages[-1][-1] = (state.messages[-1]
        #                                 [-1], edited_image_pil, "Crop")
        #     if api_name in ['grounding_dino+sam', 'grounded_sam']:
        #         edited_image_pil = Image.open(
        #             BytesIO(base64.b64decode(state.get_images()[0]))).convert("RGB")
        #         edited_image_pil = plot_boxes(edited_image_pil, tool_response)
        #         edited_image_pil = plot_masks(
        #             edited_image_pil, tool_response_clone)
        #         state.messages[-1][-1] = (state.messages[-1]
        #                                 [-1], edited_image_pil, "Crop")
        #     if api_name in ['sam']:
        #         if 'points' in tool_cfg[0]['API_params']:
        #             edited_image_pil = Image.open(
        #                 BytesIO(base64.b64decode(state.get_images()[0]))).convert("RGB")
        #             edited_image_pil = plot_masks(
        #                 edited_image_pil, tool_response_clone)
        #             tool_response_clone['points'] = tool_cfg[0]['API_params']['points']
        #             tool_response_clone['point_labels'] = tool_cfg[0]['API_params']['point_labels']
        #             edited_image_pil = plot_points(
        #                 edited_image_pil, tool_response_clone)

        #             state.messages[-1][-1] = (state.messages[-1]
        #                                     [-1], edited_image_pil, "Crop")
        #         else:
        #             assert 'boxes' in tool_cfg[0]['API_params'], "not find 'boxes' in {}".format(
        #                 tool_cfg[0]['API_params'].keys())
        #             edited_image_pil = Image.open(
        #                 BytesIO(base64.b64decode(state.get_images()[0]))).convert("RGB")
        #             edited_image_pil = plot_boxes(edited_image_pil, tool_response)
        #             tool_response_clone['boxes'] = tool_cfg[0]['API_params']['boxes']
        #             edited_image_pil = plot_masks(
        #                 edited_image_pil, tool_response_clone)
        #             state.messages[-1][-1] = (state.messages[-1]
        #                                     [-1], edited_image_pil, "Crop")

        #     yield (state, state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state)) + (enable_btn,) * 6

        # finish_tstamp = time.time()
        # logger.info(f"{output}")

        # # models = get_model_list()

        # # FIXME: disabled temporarily for image generation.
        # with open(get_conv_log_filename(), "a") as fout:
        #     data = {
        #         "tstamp": round(finish_tstamp, 4),
        #         "type": "chat",
        #         "model": model_name,
        #         "start": round(start_tstamp, 4),
        #         "finish": round(start_tstamp, 4),
        #         "state": state.dict(force_str=True),
        #         "images": all_image_hash,
        #         "ip": request.client.host,
        #     }
        #     fout.write(json.dumps(data) + "\n")
            