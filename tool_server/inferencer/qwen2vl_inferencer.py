from .base_inferencer import BaseInferencer
import uuid,requests,time
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *

inferencer_id = str(uuid.uuid4())[:6]
logger = build_logger("base_inferencer", f"base_inferencer_{inferencer_id}.log")
class QwenInferencer(BaseInferencer):
    def __init__(self, controller_addr="http://localhost:20001"):
        super().__init__(controller_addr)
    
    def model_specific_process_conversation(
        self,
        text_prompt,
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
                        "text": text_prompt
                    }
                ]
            }
        ]

        return messages
    
    
    def model_specific_append_message_to_conversation(
        self, 
        conversation, 
        text_prompt, 
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
                            "text": text_prompt
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
                            "text": text_prompt
                        }
                    ]
                }
            ]
        
        conversation.extend(new_messages)

        return conversation
    
    def get_model_response(self, model_name, conversation, image=None, gen_kwargs={}):
        assert model_name in self.available_models
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
        