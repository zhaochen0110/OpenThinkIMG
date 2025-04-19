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
# from tool_server.tf_eval.tool_inferencer.dynamic_batch_manager.dynamic_batch_manager import DynamicBatchItem, DynamicBatchManager

from .template_instruct import *
from dataclasses import dataclass, field, asdict
from typing import Dict, Sequence, Optional,List

# Set the visible CUDA device (adjust as necessary)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
    image_tool_manager: Optional[ImageToolManager] = None,
    newest_image: Optional[Image.Image] = None
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
    
    def extract_actions(text: str):
        """
        Extract only the 'actions' list from the model response text.
        
        Args:
            text (str): The model response text containing actions
            
        Returns:
            Optional[List]: The parsed actions list or None if extraction fails
        """
        try:
            # Try to find the "actions" part using regex
            actions_pattern = r'"actions"\s*:\s*(\[(?:[^\[\]]|\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])*\])'
            actions_match = re.search(actions_pattern, text)
            
            if not actions_match:
                return None
                
            actions_str = actions_match.group(1)
            actions_list = json.loads(actions_str)
            return actions_list
            
        except Exception as e:
            print(f"Error extracting actions list: {e}")
            return None
    
    if not model_response:
        return None

    try:
        if model_mode == "general":
            # pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*\}'
            # match = re.search(pattern, model_response)
            # if not match:
            #     return None
            # data = json.loads(match.group(0))

            # if "actions" not in data or not data["actions"]:
            #     return None
            actions = extract_actions(model_response)
            if not actions:
                return None

            parsed_actions = []
            for action in actions:
                # If an image (as base64) is provided in the action's arguments, process it.
                if image_tool_manager and 'image' in action.get('arguments', {}):
                    base64_img = action['arguments']['image']
                    img_key = image_tool_manager.process_base64_image(base64_img)
                    if img_key:
                        action['arguments']['image'] = img_key
                elif newest_image and 'image' in action.get('arguments', {}):
                    newest_image_base64 = pil_to_base64(newest_image, url_format=False)
                    action['arguments']['image'] = newest_image_base64

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
        print("Wrong model response:", model_response)
        return None


##############################################
# 2. Process tool call results and update the conversation prompt
##############################################

def pil_to_base64(img: Image.Image, url_format = True) -> str:
    """
    Convert a PIL image to a base64 encoded string.
    
    Args:
        img (Image.Image): The PIL image to convert.
        
    Returns:
        str: Base64 encoded string representation of the image.
    """
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    if url_format:
        img_str = f"data:image/jpeg;base64,{img_str}"
    return img_str

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
    input_data_item: Dict = None,
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
                # # Note: Assumes that image_tool_manager has an 'add_image' method.
                # image_tool_manager.add_image(edited_image)
                # Convert the base64 string to a PIL image.
                # if edited_image == None:
                #     breakpoint()
                edited_image = base64_to_pil(edited_image)
                if input_data_item:
                    input_data_item["images"].append(edited_image)
                    
            else:
                edited_image = None

            # Extract text output from the tool result.
            tool_response_text = tool_result.get("text", None)
            # Retrieve the API name from the result (supporting multiple key names)
            api_name = cfg.get("API_name", cfg.get("api_name", ""))

            # Construct a new prompt based on the model mode.
            if model_mode == "llava_plus": 
                new_response = f"{api_name} model outputs: {tool_response_text}\n\n"
                new_round_prompt = (
                    f"{new_response} Please summarize the model outputs "
                    f"and answer my first question."
                )
            elif model_mode == "general":
                new_response = f"OBSERVATION:\n{api_name} model outputs: {tool_response_text}\n"
                new_round_prompt = (
                    f"{new_response}Please summarize the model outputs "
                    f"and answer my first question."
                )

        except Exception as e:
            # In case of errors, revert to the original prompt.
            print(f"Error in handle_tool_result: {e}")
            edited_image = None
            new_round_prompt = original_prompt

    # Pop previous images since vllm only supports one image
    # if input_data_item:
    #     for conv in input_data_item["conversations"]:
    #         for idx,c in enumerate(conv["content"]):
    #             if c["type"] == "image" or c["type"] == "image_url":
    #                 del conv["content"][idx]
    # Append the new message (with text and optional image) to the conversation history.
    updated_conversations = append_conversation_fn(
        conversation=conversations, 
        text=new_round_prompt, 
        image=input_data_item["images"][-1] if input_data_item else edited_image, 
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
    role: str = "user",
    # formatting: str = "qwen"
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
        image_base64 = pil_to_base64(image)
        new_messages = [
            {
                "role": role,
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_base64}
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
        if "<stop>" in new_text:
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
# 5. Multi-turn generation with tool calls Using VLLM
##############################################

def vllm_generate_with_tool_calls(
    vllm_model,
    prompts,
    images,
    sampling_params = None,
    max_rounds: int = 3,
    model_mode: str = "general",
    controller_addr: str = "http://SH-IDCA1404-10-140-54-5:20001",
):
    """
    Perform multi-turn generation with support for tool calls. Dynamically inserts tool outputs into the conversation.
    
    Args:
        vllm_model: The VLLM model instance used for text generation
        all_multimodal_inputs: List of dictionaries containing prompts and multimodal data
            Each dict should have format: {"prompt": str, "multi_modal_data": {"image": PIL.Image}}
        sampling_params: Parameters for text generation sampling
        use_tqdm: Whether to show progress bar during generation
        max_rounds: Maximum number of conversation rounds/turns
        model_mode: Mode of operation for the model ("general" or other modes)
        
    Returns:
        tool_generation_outputs: List of dictionaries containing the tool generation outputs, including:
            - conversations: List of conversation messages
            - status: Processing status ("processing" or "finished") 
            - model_outputs: List of model generated texts
            - model_output_ids: List of model output token IDs
            - tool_cfgs: Tool configurations used
            - tool_outputs: Outputs from tool calls
            - new_round_input: New inputs for next round
            - images: List of images used
            - prompt: Original input prompt
    """
    tool_manager = ToolManager(controller_addr)
    tool_manager.available_tools = [tool for tool in tool_manager.available_tools if tool not in ['crop', 'drawline']]
    print(f"controller_addr: {controller_addr}")
    print(f"Avaliable tools are {tool_manager.available_tools}")
    miss_tool = []
    for tool in ["ZoomInSubfigure","DrawHorizontalLineByY","OCR","DrawVerticalLineByX","SegmentRegionAroundPoint","Point"]:
        if tool not in tool_manager.available_tools:
            miss_tool.append(tool)
    if len(miss_tool) == 0:
        print("All tools are called successfully")
    else:
        print(f"Not all tools is called successfully, missing tool {miss_tool}")

    # image_tool_manager = ImageToolManager()
    # {"prompt": p, "multi_modal_data": {"image": i}}
    
    ## build data

    
    input_data = []

    
    for prompt, image in zip(prompts, images):
        current_image = image
        if current_image:
            if current_image.mode in ("RGBA", "LA", "P"):
                current_image = current_image.convert("RGB") 
                    
        current_image_base64 = pil_to_base64(current_image)
        if isinstance(prompt, list):
            for p in prompt:
                for c in p["content"]:
                    if c["type"] == "image":
                        c["type"] = "image_url"
                        c["image_url"] = {"url": current_image_base64}
                        c.pop("image", None)
            initial_user_messages = prompt
            # get prompt text
            contents = prompt[-1]["content"]
            current_prompt = ""
            for content in contents:
                if content["type"] == "text" and content["text"]:
                    current_prompt += content["text"]
        elif isinstance(prompt, str):
            current_prompt = prompt
            initial_user_messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": current_image_base64}}, 
                {"type": "text", "text": current_prompt}
            ]
        }]
        else:
            raise ValueError("Prompt should be either a string or a list of messages")
        
        
        data_instance = dict(
            conversations = initial_user_messages,
            status = "processing",
            model_outputs = [],
            model_output_ids = [],
            tool_cfgs = [],
            tool_outputs = [],
            new_round_input = [],
            images = [current_image],
            prompt = current_prompt
        )
        input_data.append(data_instance)

    # breakpoint()    
    ## Vllm inference with tool calling
    for _ in range(max_rounds):
        input_conversations = [item["conversations"] for item in input_data if item["status"] == "processing"]
        input_idxs = [idx for idx, item in enumerate(input_data) if item["status"] == "processing"]
        try:
            # breakpoint()
            outputs = vllm_model.chat(
                input_conversations,
                sampling_params = sampling_params,
                use_tqdm = False,
            )
            output_texts = [output.outputs[0].text for output in outputs]
            output_idss = [output.outputs[0].token_ids for output in outputs]
        except Exception as e:
            # breakpoint()
            print(f"[vllm generation] {e}")
            output_texts = ["Model generation error"] * len(input_conversations)
            output_idss = [(1712, 9471, 1465, 151645)] * len(input_conversations)
            
        ## update data
        for input_idx, output_text, output_ids in zip(input_idxs, output_texts, output_idss):
            input_data[input_idx]["model_outputs"].append(output_text)
            input_data[input_idx]["model_output_ids"].append(output_ids)
            # Append the new message (with text and optional image) to the conversation history.
            input_data[input_idx]["conversations"] = append_conversation_fn(conversation=input_data[input_idx]["conversations"], text=output_text, role="assistant")
            
            ## pop qualified data
            if "Terminate" in output_text:
                input_data[input_idx]["status"] = "finished"
                continue
            
            # If a tool configuration is detected in the output, process it.
            tool_cfg = parse_tool_config(
                    output_text, 
                    model_mode=model_mode, 
                    image_tool_manager=None,
                    newest_image=input_data[input_idx]["images"][-1]
                )
            if not tool_cfg:
                input_data[input_idx]["conversations"] = append_conversation_fn(conversation=input_data[input_idx]["conversations"], text=input_data[input_idx]["prompt"], role="assistant")
            
            else:
                input_data[input_idx]["tool_cfgs"].append(tool_cfg)
                original_api_name = tool_cfg[0].get("API_name").lower() 
                api_params = tool_cfg[0].get("API_params", {})
                tool_name_mapping = {
                    'drawhorizontallinebyy': 'DrawHorizontalLineByY',
                    'zoominsubfigure': 'ZoomInSubfigure',
                    'drawverticallinebyx': 'DrawVerticalLineByX',
                    'segmentregionaroundpoint': 'SegmentRegionAroundPoint',
                    'point': 'Point',
                    'ocr': 'OCR'
                }
                api_name = tool_name_mapping.get(original_api_name)

                # breakpoint()
                # if "Terminate" in output_text:
                #     input_data[input_idx]["status"] = "finished"
                #     continue
                
                # print(f"Tool calling: {api_name}")
                # Call the tool using the tool manager
                # breakpoint()
                if "param" in api_params:
                    p = api_params["param"]
                    print(f"Tool name: {api_name}, params: {p}")
                tool_result = tool_manager.call_tool(api_name, api_params)
                # Append the tool call output to the conversation history
                input_data[input_idx]["tool_outputs"].append(tool_result)
                # Process the tool result and update the conversation
                input_data[input_idx]["conversations"] = handle_tool_result(
                    cfg=tool_cfg[0],
                    tool_result=tool_result, 
                    conversations=input_data[input_idx]["conversations"], 
                    model_mode="general", 
                    original_prompt=input_data[input_idx]["prompt"], 
                    input_data_item = input_data[input_idx]
                )

    output_ids = [item["model_output_ids"][-1] for item in input_data]

    return input_data



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