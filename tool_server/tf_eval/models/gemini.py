from .abstract_model import tp_model
import uuid,requests,time
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from typing import List
from qwen_vl_utils import process_vision_info
from openai import OpenAI
import os
from ..utils.utils import *
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem
from .template_instruct import *
from ..utils.log_utils import get_logger
import google.generativeai as genai

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 2048,
  "response_mime_type": "text/plain",
}


inferencer_id = str(uuid.uuid4())[:6]
logger = get_logger(__name__)
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


class GeminiModels(tp_model):
    def __init__(
      self,  
      model_name: str = None,
      max_retry: int = None
    ):
        self.model_name = model_name
        self.max_retry = max_retry

        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
            )


    def to(self, *args, **kwargs):
        pass


    def eval(self):
        pass
    
    def generate_conversation_fn(
        self,
        text,
        image, 
        role = "user",
    ):  
        # import pdb; pdb.set_trace()
        text = fs_cota + "\n" + "Question: " + text
        
        image = pil_to_base64(image)
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        },
                    },
                ],
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
            new_messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}"
                            },
                        },
                    ],
                }
            ]
        else:
            new_messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                        }
                    ],
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
        # In closed-source models, only sequential inference is supported, which means batch size must be 1.
        if len(batch) == 0:
            return None
        messages = []
        for item in batch:
            messages.append(item.conversation)

        return messages
    
    def generate(self, batch):
        if not batch or len(batch) == 0:
            return
        max_new_tokens = self.generation_config.get("max_new_tokens", 2048)
        
        inputs = self.form_input_from_dynamic_batch(batch)

        response = self.model.chat.completions.create(
            model = self.model_name, 
            messages = inputs,
            max_tokens = max_new_tokens)

        output_texts = response.choices[0].message.content.strip()
        import pdb; pdb.set_trace()

        # generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        
        # output_texts = self.processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )
        
        for item, output_text in zip(batch, output_texts):
            item.model_response.append(output_text)
            self.append_conversation_fn(
                item.conversation, output_text, None, "assistant"
            )
    
    
            
        
        
