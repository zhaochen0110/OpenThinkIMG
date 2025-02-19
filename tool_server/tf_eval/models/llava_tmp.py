from .abstract_model import tp_model
import uuid,requests,time
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from typing import List
from qwen_vl_utils import process_vision_info
from tool_server.tf_eval.models.llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
from tool_server.tf_eval.models.llava.model.builder import load_pretrained_model
from tool_server.tf_eval.models.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from tool_server.tf_eval.models.llava.conv import default_conversation
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()

from ..utils.utils import *
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem

from ..utils.log_utils import get_logger

import os
inferencer_id = str(uuid.uuid4())[:6]
logger = get_logger("llava_plus_model",)

class LLaVA_Plus(tp_model):
    def __init__(
        self,  
        pretrained : str = None,
    ):
        # convert the model type in checkpoint by force
        with open(os.path.join(pretrained,'config.json'), "r", encoding="utf-8") as f:
            config = json.load(f)
        config["model_type"] = "llava_plus"
        with open(os.path.join(pretrained,'config.json'), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        self.model_path = pretrained
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            pretrained, None, 'llava_plus_v0_7b', False, False, device='cpu')
        self.is_multimodal = True
        self.conv = default_conversation.copy()
        
    
    def generate_conversation_fn(
        self,
        text,
        image, 
        role = "user",
    ):
        image = pil_to_base64(image)
        messages = [
            {
                "role": role,
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]

        return messages
    
    
    def append_conversation_fn(
        self, 
        conversation, 
        text, 
        image, 
        role
    ):
        if image:
            new_messages = [
                {
                    "role": role,
                    "content": [
                        {
                            "type": "image",
                            "image": image
                        },
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
        else:
            new_messages = [
                {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
        
        conversation.extend(new_messages)

        return conversation
    
    def getitem_fn(self, meta_data, idx):
        item = meta_data[idx]
        image = Image.open(item["image_path"])
        text = item["text"]
        item_idx = item["idx"]
        res = dict(image=image, text=text, idx=item_idx)
        return res
    
    # def form_input_from_dynamic_batch(self, batch: List[DynamicBatchItem]):
    #     if len(batch) == 0:
    #         return None
    #     messages = []
    #     for item in batch:
    #         messages.append(item.conversation)
    #     texts = [
    #         self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    #         for msg in messages
    #     ]
    #     image_inputs, _ = process_vision_info(messages)
    #     inputs = self.processor(
    #         text=texts,
    #         images=image_inputs,
    #         padding=True,
    #         return_tensors="pt",
    #     )
    #     inputs = inputs.to(self.model.device)
    #     return inputs
    
    def generate(self, batch):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor
        messages = []
        for item in batch:
            messages.append(item.conversation)
        breakpoint()
        prompt = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages
        ]
        images, _ = process_vision_info(messages)
        # prompt = batch # prompt of batch
        # images = batch # image of batch
        ori_prompt = prompt
        num_image_tokens = 0
        
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                images = [load_image_from_base64(image) for image in images]
                images = process_images(images, image_processor, model.config)

                if type(images) is list:
                    images = [image.to(self.model.device, dtype=self.model.dtype) for image in images]
                else:
                    images = images.to(self.model.device, dtype=self.model.dtype)

                replace_token = DEFAULT_IMAGE_TOKEN
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
            else:
                images = None
            image_args = {"images": images}
        else:
            images = None
            image_args = {}
        
        temperature = float(1.0)
        top_p = float(1.0)
        max_context_length = 2048
        max_new_tokens = 1024
        stop_str = None 
        do_sample = temperature > 0.001

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        if max_new_tokens < 1:
            return {
                "text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.",
                "error_code": 0
            }

        # Generate text
        generated_ids = model.generate(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        )
        
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        for item, output_text in zip(batch, generated_text):
            item.model_response.append(output_text)
            self.append_conversation_fn(
                item.conversation, output_text, None, "assistant"
            )
        
        # inputs = self.form_input_from_dynamic_batch(batch)
        # if not batch or len(batch) == 0:
        #     return
        # max_new_tokens = self.generation_config.get("max_new_tokens", 2048)
        
        # inputs = self.form_input_from_dynamic_batch(batch)
        # generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        
        # output_texts = self.processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )
        
        # for item, output_text in zip(batch, output_texts):
        #     item.model_response.append(output_text)
        #     self.append_conversation_fn(
        #         item.conversation, output_text, None, "assistant"
        #     )
    
    
            
        
        
