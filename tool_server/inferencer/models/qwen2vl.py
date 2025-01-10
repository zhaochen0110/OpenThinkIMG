from .abstract_model import tp_model
import uuid,requests,time
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info



from tool_server.utils.utils import *
from tool_server.utils.server_utils import *

inferencer_id = str(uuid.uuid4())[:6]
logger = build_logger("qwen2vl_model", f"qwen2vl_model_{inferencer_id}.log")

class Qwen2VL(tp_model):
    def __init__(
      self,  
      model_path : str = None,
    ):
        self.model_path = model_path
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="cpu"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
    
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
    
    
    def form_input_from_dynamic_batch(self, batch):
        messages = []
        for item in batch:
            messages.extend(item.conversation)
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
    
    def generate(self, batch, gen_kwargs={}):
        max_new_tokens = gen_kwargs.get("max_new_tokens", 2048)
        
        inputs = self.form_input_from_dynamic_batch(batch)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        for item, output_text in zip(batch, output_texts):
            item.model_response.append(output_text)
            self.append_conversation_fn(
                item.conversation, output_text, None, "assistant"
            )
            
        
        
    
    def get_model_response(self, model_name, conversation, image=None, gen_kwargs={}):
        assert model_name in self.available_models, f"model_name {model_name} not in available models, {self.available_models}"
        assert isinstance(gen_kwargs, dict)
        
        pload = {
            "model": model_name,
            "conversation": conversation,
            "temperature": gen_kwargs.get("temperature", 0.0),
            "top_p": gen_kwargs.get("top_p", 1.0),
            "max_new_tokens": gen_kwargs.get("max_new_tokens", 2048),
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
                        output = data["text"]
                        logger.debug(output, end="\0", flush=True)
                    else:
                        output = data["text"] + \
                            f" (error_code: {data['error_code']})"
                        return
                    time.sleep(0.03)
        except requests.exceptions.RequestException as e:
            logger.error(f"error: {e}")
            return
        
        logger.info(f"==== response ====")
        logger.info(output)
        
        return output
        