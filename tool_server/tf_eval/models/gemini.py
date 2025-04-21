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
import time
import random

inferencer_id = str(uuid.uuid4())[:6]
logger = get_logger(__name__)
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


class GeminiModels(tp_model):
    def __init__(
      self,  
      model_name: str = None,
      max_retry: int = None,
      temperature: float = None
    ):
        self.model_name = model_name
        self.max_retry = max_retry
        self.temperature = temperature

        # self.model = genai.GenerativeModel(
        #     model_name="gemini-2.0-flash-exp",
        #     generation_config=generation_config,
        #     )

        self.api_keys = []
        self.api_key = random.choice(self.api_keys)
        self.model = OpenAI(
            api_key=self.api_key,  # Google Gemini API key
            base_url="https://generativelanguage.googleapis.com/v1beta/"  # Gemini base URL
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
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": online_update_system_prompt}],
            }
        ]

        # Add FS examples to the conversation
        for fs in fs_example_offlinetype:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": fs["user_request"]}],
            })

            assistant_reply = []
            for step in fs['steps']:


                # Combine thought and actions as assistant's response
                step_content = {
                    "thought": step["thought"],
                    "actions": step["actions"],
                }

                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": json.dumps(step_content)}],
                })

                if step["observation"] != {}:
                    # Adding the observation as assistant's content
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": "OBSERVATION:\n" + step["observation"]}],
                    })
                # breakpoint()

        # Add text and image for the current conversation
        image_base64 = pil_to_base64(image)  # Convert image to base64 string
        messages.append(
            {
                "role": role,
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }
        )
        # breakpoint()
        return messages
    
    
    def append_conversation_fn(
        self, 
        conversation, 
        text, 
        image, 
        role
    ):
        # Prepare text content
        new_content = [
            {
                "type": "text",
                "text": text,
            }
        ]

        # Add image(s) if provided
        if image:
            if isinstance(image, list):  # Handle multiple images
                for img in image:
                    if isinstance(img, str):
                        new_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img}"
                            }
                        })
                    else:
                        raise ValueError("List elements must be strings")
            elif isinstance(image, str):  # Handle a single image
                new_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                })
            else:
                raise ValueError("Image must be a string or a list of strings")

        # Create new message entry
        new_messages = [
            {
                "role": role,
                "content": new_content,
            }
        ]

        # Add new messages to the conversation
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
        fail_times = 1
        fail_flag = False
        base_sleeptime = 15
        
        # breakpoint()

        while fail_times < int(self.max_retry):
            try:
                response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=inputs[0],
                    max_tokens=max_new_tokens,
                    temperature=self.temperature
                )
                final_response = response.choices[0].message.content.strip()

                output_texts = [final_response]

                for item, output_text in zip(batch, output_texts):
                    item.model_response.append(output_text)
                    self.append_conversation_fn(
                        item.conversation, output_text, None, "assistant"
                    )

                # 如果成功，直接退出循环
                fail_flag = False
                break


            except Exception as e:
                new_api = random.choice(self.api_keys)
                while new_api == self.api_key:
                    new_api = random.choice(self.api_keys)
                self.api_key = new_api
                self.model = OpenAI(
            api_key=new_api,
            base_url="https://generativelanguage.googleapis.com/v1beta/"  # Gemini base URL
                )
                logger.error(
                    f"Error: {e}, retrying in {fail_times * base_sleeptime} seconds"
                )
                fail_times += 1
                fail_flag = True
                time.sleep(fail_times * base_sleeptime)

        if fail_flag:
            logger.error(f"Failed to generate response after {self.max_retry} attempts")
            
        
        
