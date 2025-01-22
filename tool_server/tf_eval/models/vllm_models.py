from .abstract_model import tp_model
import uuid,requests,time
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from typing import List
from vllm import LLM, SamplingParams


from .template_instruct import *
from ..utils.utils import *
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem

from ..utils.log_utils import get_logger
inferencer_id = str(uuid.uuid4())[:6]
logger = get_logger("vllm_models",)

class VllmModels(tp_model):
    def __init__(
      self,  
      pretrained : str = None,
      tensor_parallel: str = "1",
    ):
        tensor_parallel = int(tensor_parallel)  
        self.model = LLM(
            model=pretrained,
            tensor_parallel_size=tensor_parallel,
        )

    def generate_conversation_fn(
        self,
        text,
        image, 
        role = "user",
    ):  
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
        sampling_params = SamplingParams(max_tokens=max_new_tokens)
        
        inputs = self.form_input_from_dynamic_batch(batch)
        response = self.model.chat(inputs, sampling_params)
        
        # import debugpy
        # debugpy.listen(address = ('0.0.0.0', 7119))
        # debugpy.wait_for_client() 
        # breakpoint() # 在下一句代码处暂停
        # dist.barrier()
        for item, output_item in zip(batch, response):
            output_text = output_item.outputs[0].text
            item.model_response.append(output_text)
            self.append_conversation_fn(
                item.conversation, output_text, None, "assistant"
            )
    
    def to(self, *args, **kwargs):
        return self
    
    def eval(self):
        return self
            
        
        
