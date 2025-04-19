import base64
import io
from io import BytesIO  # For handling byte streams
import torch
import re
import os
from PIL import Image
import json
from typing import Optional, Dict, List
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoTokenizer, 
    AutoProcessor, 
    GenerationConfig
)
from qwen_vl_utils import process_vision_info
from accelerate import Accelerator
from trl.models import unwrap_model_for_generation
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from .template_instruct import *

# Set the visible CUDA device (adjust as necessary)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ImageToolManager:
    """
    Manager to handle image storage and conversion for tool calls.
    """
    def __init__(self):
        self.image_dict = {}
        self.current_img_index = 1

    def add_initial_image(self, image: Image.Image) -> str:
        """
        Convert the initial image to a base64 string and store it in the dictionary.
        
        Args:
            image (Image.Image): The initial PIL image.
            
        Returns:
            str: The key for the stored image.
        """
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        self.image_dict['img_1'] = img_base64
        return 'img_1'

    def process_base64_image(self, base64_str: str) -> Optional[str]:
        """
        Decode a base64 string into a PIL image and store it in the dictionary.
        
        Args:
            base64_str (str): Base64 encoded image string.
        
        Returns:
            Optional[str]: The image key if processing succeeds; otherwise, None.
        """
        try:
            # Decode the base64 string into image data
            image_data = base64.b64decode(base64_str)
            # Convert the decoded bytes into a PIL image
            image = Image.open(io.BytesIO(image_data))
            
            # Increment the image index and create a new key
            self.current_img_index += 1
            new_img_key = f'img_{self.current_img_index}'
            
            # Store the image in the dictionary
            self.image_dict[new_img_key] = image
            return new_img_key
        except Exception as e:
            print(f"Error processing base64 image: {e}")
            return None

    def store_tool_image(self, base64_str: str) -> Optional[str]:
        """
        Decode a base64 string into a PIL image and store it.
        (Note: This function does not currently return the new image key on success.)
        
        Args:
            base64_str (str): Base64 encoded image string.
        
        Returns:
            Optional[str]: The image key if processing succeeds; otherwise, None.
        """
        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))
            
            self.current_img_index += 1
            new_img_key = f'img_{self.current_img_index}'
            self.image_dict[new_img_key] = image
            
            # Optionally, you might want to return new_img_key here.
            return new_img_key
        except Exception as e:
            print(f"Error processing base64 image: {e}")
            return None

    def get_image_by_key(self, img_key: str) -> Optional[Image.Image]:
        """
        Retrieve an image from the storage by its key.
        
        Args:
            img_key (str): The key corresponding to the stored image.
            
        Returns:
            Optional[Image.Image]: The retrieved PIL image, or None if not found.
        """
        return self.image_dict.get(img_key)


##############################################
# 1. Functions for detecting and parsing tool call configurations
##############################################

def detect_tool_config(model_response: str, model_mode: str = "general") -> bool:
    """
    Detect whether the model response contains a tool call configuration.
    
    Args:
        model_response (str): The output string from the model.
        model_mode (str, optional): The mode for detection ('general' or 'llava_plus'). Defaults to "general".
        
    Returns:
        bool: True if a tool configuration is detected; otherwise, False.
    """
    if not model_response:
        return False

    if model_mode == "general":
        # Regex to match nested JSON-like structure
        pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*\}'
        match = re.search(pattern, model_response)
        if match:
            content = match.group(0)
            return '"actions"' in content
        return False

    elif model_mode == "llava_plus":
        # Pattern specific to the "llava_plus" mode
        pattern = r'"thoughts🤔"(.*)"actions🚀"(.*)"value👉"(.*)'
        return bool(re.search(pattern, model_response, re.DOTALL))

    return False


def parse_tool_config(
    model_response: str, 
    model_mode: str = "general", 
    image_tool_manager: Optional[ImageToolManager] = None
) -> Optional[List[Dict]]:
    """
    Parse the tool configuration from the model response and handle image conversion if necessary.
    
    Args:
        model_response (str): The model's generated response.
        model_mode (str): The mode for parsing.
        image_tool_manager (Optional[ImageToolManager]): Manager for image conversion and storage.
        
    Returns:
        Optional[List[Dict]]: A list of parsed tool configurations, or None if parsing fails.
    """
    if not model_response:
        return None

    try:
        if model_mode == "general":
            pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*\}'
            match = re.search(pattern, model_response)
            if not match:
                return None
            data = json.loads(match.group(0))

            if "actions" not in data or not data["actions"]:
                return None

            parsed_actions = []
            for action in data["actions"]:
                # If an image (as base64) is provided in the action's arguments, process it.
                if image_tool_manager and 'image' in action.get('arguments', {}):
                    base64_img = action['arguments']['image']
                    img_key = image_tool_manager.process_base64_image(base64_img)
                    if img_key:
                        action['arguments']['image'] = img_key

                parsed_actions.append({
                    "API_name": action["name"],
                    "API_params": action["arguments"]
                })

            return parsed_actions

        elif model_mode == "llava_plus":
            pattern = r'"thoughts🤔"(.*)"actions🚀"(.*)"value👉"(.*)'
            matches = re.findall(pattern, model_response, re.DOTALL)
            if not matches:
                return None
            # Extract the actions part and load it as JSON (ensure valid quotes)
            actions_str = matches[0][1].strip()
            return json.loads(actions_str.replace("'", "\""))

    except Exception as e:
        print(f"[parse_tool_config] Error: {e}")
        return None


##############################################
# 2. Process tool call results and update the conversation prompt
##############################################

def base64_to_pil(b64_str: str) -> Image.Image:
    """
    Convert a base64 encoded string into a PIL image.
    
    Args:
        b64_str (str): The base64 encoded image string.
        
    Returns:
        Image.Image: The resulting PIL image.
    """
    # Remove the data URI scheme if present
    if b64_str.startswith("data:image"):
        b64_str = b64_str.split("base64,")[-1]
    return load_image_from_base64(b64_str)


def load_image_from_base64(image_str: str) -> Image.Image:
    """
    Helper function to load a PIL image from a base64 string.
    
    Args:
        image_str (str): Base64 encoded image data.
        
    Returns:
        Image.Image: The PIL image.
    """
    return Image.open(BytesIO(base64.b64decode(image_str)))


def handle_tool_result(
    cfg,
    tool_result,
    conversations,
    model_mode: str = "general",
    original_prompt: Optional[str] = None,
    image_tool_manager: Optional[ImageToolManager] = None
):
    """
    Process the tool result, update the conversation history, and generate a new prompt.
    
    Args:
        tool_result: The result returned by the tool call.
        conversations: The current conversation history.
        model_mode (str): The mode for generating the updated prompt.
        original_prompt (Optional[str]): The original user prompt.
        image_tool_manager (Optional[ImageToolManager]): Manager for handling image conversions.
        
    Returns:
        The updated conversation history.
    """
    edited_image = None
    new_round_prompt = original_prompt

    if tool_result is not None:
        try:
            # Process image editing if an "edited_image" is provided in the tool result.
            if "edited_image" in tool_result:
                # Remove the edited image from the result and add it via the image manager.
                edited_image = tool_result.pop("edited_image")
                # Note: Assumes that image_tool_manager has an 'add_image' method.
                image_tool_manager.add_image(edited_image)
                # Convert the base64 string to a PIL image.
                edited_image = base64_to_pil(edited_image)
            else:
                edited_image = None

            # Extract text output from the tool result.
            tool_response_text = tool_result.get("text", None)
            # Retrieve the API name from the result (supporting multiple key names)
            api_name = cfg.get("API_name", cfg.get("api_name", ""))
            # breakpoint()
            # Construct a new prompt based on the model mode.
            if model_mode == "llava_plus": 
                new_response = f"{api_name} model outputs: {tool_response_text}\n\n"
                new_round_prompt = (
                    f"{new_response} Please summarize the model outputs "
                    f"and answer my first question"
                )
            elif model_mode == "general":
                new_response = f"OBSERVATION:\n{api_name} model outputs: {tool_response_text}\n"
                new_round_prompt = (
                    f"<tool_result>{new_response}Please summarize the model outputs "
                    f"and answer my first question</tool_result>"
                )

        except Exception as e:
            # In case of errors, revert to the original prompt.
            print(f"Error in handle_tool_result: {e}")
            edited_image = None
            new_round_prompt = original_prompt

    # Append the new message (with text and optional image) to the conversation history.
    updated_conversations = append_conversation_fn(
        conversation=conversations, 
        text=new_round_prompt, 
        image=edited_image, 
        role="user"
    )

    return updated_conversations


##############################################
# 3. Token-level conversation management
##############################################

def generate_conversation_fn(
    model,
    current_prompt_inputs: dict,
    tokenizer,
    generation_config
):
    """
    Generate a small step in the conversation. Returns the new text generated and the full output tensor.
    
    Args:
        model: The generation model.
        current_prompt_inputs (dict): The current prompt inputs.
        tokenizer: The tokenizer for decoding generated tokens.
        generation_config: Generation configuration parameters.
        
    Returns:
        Tuple containing the newly generated text and the full output tensor.
    """
    full_outputs = model.generate(
        **current_prompt_inputs,
        **generation_config.to_dict()
    )

    # Calculate the number of new tokens generated
    old_length = current_prompt_inputs["input_ids"].shape[1]
    new_tokens = full_outputs[:, old_length:]

    # Decode the new tokens into text
    new_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    return new_text, full_outputs


def append_conversation_fn(
    conversation, 
    text: str, 
    image=None, 
    role: str = "user"
):
    """
    Append a new message to the conversation history.
    
    Args:
        conversation (list): The current conversation history.
        text (str): The text message to append.
        image: (Optional) An image to include with the message.
        role (str): The role of the sender (default is "user").
        
    Returns:
        The updated conversation list.
    """
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


def convert_conversation_to_prompt_inputs(
    conversations, 
    processor, 
    model, 
    add_generation_prompt=True
):
    """
    Convert the conversation history into a format acceptable by the model.
    
    Args:
        conversations (List[Dict]): The conversation history.
        processor: The processor to handle text templates and images.
        model: The model (used to determine the target device).
        add_generation_prompt (bool, optional): Whether to add a generation prompt. Defaults to True.
        
    Returns:
        Dict: The prompt inputs that can be directly passed to the generation function.
    """
    # Apply the chat template for each message in the conversation
    texts = [
        processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=add_generation_prompt) 
    ]
    
    # Process any image inputs from the conversation
    image_inputs, _ = process_vision_info(conversations)
    # breakpoint() 
    # Create the model inputs using the processor
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # Move inputs to the model's device
    inputs = inputs.to(model.device)
    return inputs


##############################################
# 4. Multi-turn generation with tool calls
##############################################

def generate_with_tool_calls(
    model,
    prompt_inputs: dict,
    generation_config,
    processor,
    tokenizer,
    question: str,  # The original user question
    max_rounds: int = 3,
    model_mode: str = "general",
    initial_image: Optional[Image.Image] = None,
    initial_prompts: Optional[str] = None
):
    """
    Perform multi-turn generation with support for tool calls. Dynamically inserts tool outputs into the conversation.
    
    Args:
        model: The generation model.
        prompt_inputs (dict): The initial prompt inputs.
        generation_config: Configuration for generation.
        processor: Processor for handling text and image inputs.
        tokenizer: Tokenizer for decoding outputs.
        question (str): The original user question.
        max_rounds (int, optional): Maximum number of generation rounds. Defaults to 3.
        model_mode (str, optional): Model mode ("general" or "llava_plus"). Defaults to "general".
        initial_image (Optional[Image.Image]): An optional initial image.
        initial_prompts (Optional[str]): An optional initial prompt text.
        
    Returns:
        Tuple: The final generated outputs and the image tool manager.
    """
    tool_manager = ToolManager()
    image_tool_manager = ImageToolManager()
    if initial_image.mode in ("RGBA", "LA", "P"):
        initial_image = initial_image.convert("RGB") 
    # Add the initial image to the image manager if provided
    if initial_image:
        image_tool_manager.add_initial_image(initial_image)
    
    conversations = []
    # Create the initial user message containing both image and text
    initial_user_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": initial_image}, 
            {"type": "text", "text": initial_prompts}
        ]
    }
    conversations.append(initial_user_message)

    final_outputs = None

    # Loop for a maximum number of rounds to generate responses and process tool calls.
    for _ in range(max_rounds):
        new_text, full_outputs = generate_conversation_fn(
            model=model,
            current_prompt_inputs=prompt_inputs,
            tokenizer=tokenizer,
            generation_config=generation_config
        )
        final_outputs = full_outputs
        if "Terminate" in new_text:
            return full_outputs 

        # Stop generation if a stop token is detected
        if "<stop>" in new_text or tokenizer.eos_token in new_text:
            break
        
        # If a tool configuration is detected in the output, process it.
        if detect_tool_config(new_text, model_mode=model_mode):
            tool_cfg = parse_tool_config(
                new_text, 
                model_mode=model_mode, 
                image_tool_manager=image_tool_manager
            )
            if not tool_cfg:
                continue


            api_name = tool_cfg[0].get("API_name")
            api_params = tool_cfg[0].get("API_params", {})

            # If the tool call requires an image, retrieve it from the image manager.
            if 'image' in api_params:
                img_key = api_params['image']
                api_params['image'] = image_tool_manager.get_image_by_key(img_key)

            # Call the tool using the tool manager
            tool_result = tool_manager.call_tool(api_name, api_params)
            # Append the tool call output to the conversation history
            conversations = append_conversation_fn(conversation=conversations, text=new_text, role="assistant")
            # Process the tool result and update the conversation
            update_conversation = handle_tool_result(
                cfg=tool_cfg[0],
                tool_result=tool_result, 
                conversations=conversations, 
                model_mode="general", 
                original_prompt=question, 
                image_tool_manager=image_tool_manager
            )
            # Convert the updated conversation to prompt inputs for the next round
            prompt_inputs = convert_conversation_to_prompt_inputs(
                conversations=update_conversation, 
                processor=processor, 
                model=model
            )

    return final_outputs


##############################################
# Main execution
##############################################

# if __name__ == "__main__":
#     # Define the model name or path and load the model.
#     model_name = "/mnt/petrelfs/share_data/songmingyang/model/mm/Qwen2-VL-2B-Instruct"
#     model = Qwen2VLForConditionalGeneration.from_pretrained(
#         model_name, torch_dtype="auto", device_map="auto"
#     )

#     # Load the processor for both text and image inputs.
#     processor = AutoProcessor.from_pretrained(model_name)


#     initial_prompt = "What is the average Operating profit of the H&M Group worldwide from 2019 to 2020?"
#     image_path = "/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/processed_data_chartqa/images/chartcot_0/1.jpeg" 

#     prompt = online_fs_cota + "\n" + "Question: " + initial_prompt
#     image = Image.open(image_path)
    
  
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image"},  # The actual image will be handled via the processor
#                 {"type": "text", "text": prompt},
#             ],
#         }
#     ]

#     # Process the conversation template to produce text input
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )

#     # Create the model inputs from text and image
#     inputs = processor(
#         text=[text],
#         images=[image],
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to("cuda")

#     # Set the generation configuration parameters
#     generation_config = GenerationConfig(
#         max_new_tokens=100,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.9
#     )

#     # Use Accelerator to prepare the model (for distributed or mixed precision training/inference)
#     accelerator = Accelerator()
#     model = accelerator.prepare(model)

#     # Run the model with tool call integration within the unwrap context manager
#     with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
#         final_outputs = generate_with_tool_calls(
#             model=unwrapped_model,
#             prompt_inputs=inputs,
#             generation_config=generation_config,
#             processor=processor,
#             tokenizer=processor.tokenizer,
#             max_rounds=3,
#             model_mode="general",
#             question=initial_prompt,
#             initial_image=image,
#             initial_prompts=prompt
#         )

#     # Decode the final generated tokens into text
#     decoded_output = processor.tokenizer.decode(final_outputs[0], skip_special_tokens=True)
#     print("\nFinal Outputs:\n", final_outputs)
#     print("\nDecoded Final Output:\n", decoded_output)