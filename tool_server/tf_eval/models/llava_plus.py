from .abstract_model import tp_model
import uuid,requests,time
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from typing import List
import torch

from tool_server.tf_eval.models.llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
from tool_server.tf_eval.models.llava.constants import IMAGE_TOKEN_INDEX
from tool_server.tf_eval.utils.utils import pil_to_base64
from tool_server.tf_eval.models.llava.model.builder import load_pretrained_model

from ..utils.utils import *
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem

from ..utils.log_utils import get_logger
from transformers import logging as tf_logging

tf_logging.set_verbosity_error()

inferencer_id = str(uuid.uuid4())[:6]
logger = get_logger("llava_plus_model",)

class LLaVA_Plus(tp_model):
    def __init__(
        self,  
        pretrained : str = None,
    ):
        self.model_path = pretrained
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(pretrained, None, 'llava_plus_v0_7b', False, False, device='cpu')
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        
        # Rename the model type from 'llava' -> 'llava_plus' to support Transformers > 4.31.1
        with open(os.path.join(pretrained,'config.json'), "r", encoding="utf-8") as f:
            config = json.load(f)
        config["model_type"] = "llava_plus"
        with open(os.path.join(pretrained,'config.json'), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def generate_conversation_fn(
        self,
        text,
        image, 
        role = "user",
    ):
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
    
    def form_input_from_dynamic_batch(self, batch: List[DynamicBatchItem]):
        if len(batch) == 0:
            return None
        messages = []
        for item in batch:
            messages.append(item.conversation)
        prompts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        input_ids = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(self.model.device) for prompt in prompts]
        image_batch = []
        for msg in messages:
            all_images = []
            for item in msg:
                content = item['content']
                for dic in content:
                    if dic['type'] == 'image':
                        all_images.append(dic['image'])
            all_images = [load_image_from_base64(pil_to_base64(image)) for image in all_images]
            all_images = process_images(all_images, self.image_processor, self.model.config)
            image_batch.append(all_images.to(self.model.device, dtype=self.model.dtype))
        
        inputs = {
            'input_ids': input_ids,
            'images': image_batch
        }
        return inputs
    
    def generate(self, batch):
        if not batch or len(batch) == 0:
            return
        max_new_tokens = self.generation_config.get("max_new_tokens", 2048)
        
        inputs = self.form_input_from_dynamic_batch(batch)
        
        generated_ids_trimmed = []
        input_ids = inputs['input_ids']
        images = inputs['images']
        for input_ids, images in zip(inputs['input_ids'], inputs['images']):
            input_ids = input_ids.unsqueeze(0)
            stopping_criteria = KeywordsStoppingCriteria(['\n###'], self.tokenizer, input_ids)
            stopping_criteria.keyword_ids = [torch.Tensor([13, 2277, 29937])]
            cur_round_max_new_tokens = max_new_tokens - len(input_ids) - 1
            generated_ids = self.model.generate(input_ids=input_ids, images=images, 
                                                eos_token_id=self.tokenizer.eos_token_id,
                                                pad_token_id=self.tokenizer.pad_token_id,
                                                stopping_criteria=[stopping_criteria],
                                                max_new_tokens=cur_round_max_new_tokens,
                                                do_sample=True,
                                                temperature=0.8
                                                )[0]
            generated_ids[generated_ids == -200] = self.tokenizer.pad_token_id
            generated_ids_trimmed.append(generated_ids[len(input_ids):])
        
        output_texts = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        for item, output_text in zip(batch, output_texts):
            item.model_response.append(output_text)
            self.append_conversation_fn(
                item.conversation, output_text, None, "assistant"
            )
