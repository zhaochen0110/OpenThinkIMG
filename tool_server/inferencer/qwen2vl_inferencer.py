from base_inferencer import BaseInferencer

class QwenInferencer(BaseInferencer):
    def __init__(self, controller_addr="http://localhost:20001"):
        super().__init__(controller_addr)
    
    def model_specific_process_conversation(self, text_prompt, image, role):
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
    
    
    def model_specific_append_message_to_conversation(self, old_messages, text_prompt, image, role):
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
        
        messages = old_messages.extend(new_messages)

        return messages
        