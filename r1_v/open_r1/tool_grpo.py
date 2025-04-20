# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed

from math_verify import parse, verify
# from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from r1_v.open_r1.trainer.tool_generation import parse_tool_config
from r1_v.open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOToolTrainer, Qwen2VLGRPOToolVLLMTrainer

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy"], # "format","accuracy"
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    use_tool: Optional[bool]  = field(
        default=False,
        metadata={"help": "Whether to use tool trainer for training"},
    )
    query_key: Optional[str] = field(
        default="question",
    )
    controller_addr: Optional[str] = field(
        default="http://SH-IDCA1404-10-140-54-5:20001",
        metadata={"help": "Address of the controller"},
    )


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""


    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    output_texts = kwargs.get("model_output_texts", None)
    
    if output_texts:
        renewed_contents = []
        for output_text in output_texts:
            renewed_contents.append(output_text[-1])
    else:
        renewed_contents = contents
        
    for item, sol in zip(output_texts, solution):
        content = item[-1] 
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                tool_cfg = parse_tool_config(content)
                if tool_cfg:
                    student_answer = tool_cfg[0].get("API_params", {}).get("ans", content)
                else:
                    student_answer = content
                # content_match = re.search(r'<answer>(.*?)</answer>', content)
                # student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                if student_answer == ground_truth or float(verify(parse(student_answer), parse(ground_truth))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {item}\n")
                f.write(f"Solution: {sol}\n")
        # breakpoint()
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format, in this case, if it contains the word 'Terminate'."""
    # breakpoint()
    output_texts = kwargs.get("model_output_texts", None)
    if not output_texts:
        pattern = r'.*Terminate.*'
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
    else:
        rewards = []
        pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*\}'
        for output_text in output_texts:
            current_rewards = []
            for output_text_item in output_text:
                reward = 0.0
                try:
                    match = re.search(pattern, output_text_item)
                    assert match is not None
                    data = json.loads(match.group(0))
                    assert "thought" in data or "thoughts" in data or "actions" in data
                    reward = 1.0
                except Exception:
                    reward = 0.0
                current_rewards.append(reward)
            current_reward = sum(current_rewards) / len(current_rewards) if current_rewards else 0.0
            rewards.append(current_reward)
        return rewards
        


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
#     "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
#     "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
#     "<think> reasoning process here </think><answer> answer here </answer>"
# )
# SYSTEM_PROMPT = (
#     "[BEGIN OF GOAL] You are a visual assistant capable of generating and solving steps for chart-based reasoning. Your goal is to answer chart-related questions. You can rely on your own capabilities or use external tools to assist in solving. The available actions include: OCR, Point, DrawHorizontalLineByY, DrawVerticalLineByX, ZoomInSubfigure, and SegmentRegionAroundPoint. [END OF GOAL] \n\n"
# )

# SYSTEM_PROMPT = """You are a visual assistant capable of generating and solving steps for chart-based reasoning. Your goal is to answer chart-related questions. You can rely on your own capabilities or use external tools to assist in solving. The available actions include: OCR, Point, DrawHorizontalLineByY, DrawVerticalLineByX, ZoomInSubfigure, and SegmentRegionAroundPoint.
# Your output should be in a strict JSON format as follows:
# {"thought": "the reasoning process", "actions": [{"name": "action", "arguments": {"argument1": "value1", "argument2": "value2"}}]}
# """

SYSTEM_PROMPT = """You are a visual assistant capable of generating and solving steps for chart-based reasoning. Your goal is to answer chart-related questions. You can rely on your own capabilities or use external tools to assist in solving. Here are the available actions:
- **OCR**: Extracts text from an image. Example: `{"name": "OCR", "arguments": {"image": "img_1"}}`
- **Point**: Identifies a point in the image based on description and returns coordinates. Example: `{"name": "Point", "arguments": {"image": "img_1", "param": "x-axis value 1970"}}`
- **ZoomInSubfigure**: Crops the image to the specified subfigure. Example: `{"name": "ZoomInSubfigure", "arguments": {"image": "img_1", "param": "Downstream vs. Concept: Toy"}}`
- **SegmentRegionAroundPoint**: Segments a region around a given point. Example: `{"name": "SegmentRegionAroundPoint", "arguments": {"image": "img_1", "param": "x=\"21.5\" y=\"28.5\""}}`
- **DrawHorizontalLineByY**: Draws a horizontal line at a given y-coordinate. Example: `{"name": "DrawHorizontalLineByY", "arguments": {"image": "img_1", "param": "y=28.5"}}`
- **DrawVerticalLineByX**: Draws a vertical line at a given x-coordinate. Example: `{"name": "DrawVerticalLineByX", "arguments": {"image": "img_1", "param": "x=21.5"}}`
- **Terminate**: Ends the task and provides the final answer. Example: `{"name": "Terminate", "arguments": {"ans": "1985"}}`

To solve the problem:
1. Select actions from the provided tools list, combining them logically and building on previous steps. Call one action at a time, using its output for the next.
2. To use `SegmentRegionAroundPoint`, `DrawHorizontalLineByY`, or `DrawVerticalLineByX`, first call "Point" to get coordinates for further actions.

Your output should be in a strict JSON format as follows:
{"thought": "the reasoning process", "actions": [{"name": "action", "arguments": {"argument1": "value1", "argument2": "value2"}}]}
"""

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = load_dataset('json', data_files=script_args.dataset_name)
    # Load the image from the dataset
    def load_image_from_path(example):
        if "solution" not in example:
            example["solution"] = example["label"]
        if "label" in example:
            example.pop("label", None)

        image = Image.open(example["image_path"])  
        image = image.convert("RGBA") 
        example["image"] = image  
        return example

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example[script_args.query_key]},
            ],
        }



    QUESTION_TEMPLATE = "{Question}"

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example[script_args.query_key])},
                    ],
                },
            ],
        }


    if "image_path" in dataset[script_args.dataset_train_split].features:
        print("image in dataset")
        dataset = dataset.map(load_image_from_path)
        dataset = dataset.map(make_conversation_image)
    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("query")

    # breakpoint()
    
    if script_args.use_tool:
        trainer_cls = Qwen2VLGRPOToolTrainer if not training_args.use_vllm else Qwen2VLGRPOToolVLLMTrainer
        trainer = trainer_cls(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args),
            attn_implementation=model_args.attn_implementation,
            max_pixels=script_args.max_pixels,
            min_pixels=script_args.min_pixels,
            controller_addr=script_args.controller_addr,
        )
    else:
        trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
        trainer = trainer_cls(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args),
            attn_implementation=model_args.attn_implementation,
            max_pixels=script_args.max_pixels,
            min_pixels=script_args.min_pixels,
        )
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer


    # Train and push the model to the Hub
    trainer.train()

    # Su: add the train from the saved checkpoint

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
