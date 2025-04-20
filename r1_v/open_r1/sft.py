import logging
import os
import sys

import datasets
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from qwen_vl_utils import process_vision_info
import wandb

from transformers import TrainerCallback
from deepspeed.accelerator import get_accelerator

class EmptyCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        get_accelerator().empty_cache()
        
wandb.init(mode="disabled")
logger = logging.getLogger(__name__)

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

@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})



processor = None


def convert_example(example):
    """
    correct example into "messages" 
    eg:
    {
      "system": "You are a helpful assistant.",
      "conversations": [
          {"from": "user", "value": "How many objects are included in this image?",
           "image_path": "/path/to/image.png"},
          {"from": "assistant", "value": "<think>\nI can see 10 objects\n</think>\n<answer>\n10\n</answer>"}
      ]
    }
    """
    messages = []
    
    SYSTEM_PROMPT = ("""You are a visual assistant capable of generating and solving steps for chart-based reasoning. Your goal is to answer chart-related questions. You can rely on your own capabilities or use external tools to assist in solving. Here are the available actions:
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
    """)

    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
    })

    conversations = example.get('conversations')
    image_paths = example.get('images')
    img_idx = 0
    
    for item in conversations:
        content = []
        if item['from'] == 'human':
            role = 'user'
        else:
            role = 'assistant'
        if "You are a visual assistant capable of generating and solving steps" in item['value']:
            content.append({'type':'text', 'text':item['value'].split("\n\nQuestion: ")[-1]})
        else:
            content.append({'type':'text', 'text':item['value']})
        
        if '<image>' in item['value']:
            content.append({'type':'image', 'image':image_paths[img_idx]})
            img_idx += 1
        
        messages.append({
            'role': role,
            'content': content
        })
    
    example["messages"] = messages

    return example


def collate_fn(examples):
    texts = [processor.apply_chat_template(convert_example(example)["messages"], tokenize=False, add_generation_prompt=True) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels
    # 释放中间变量
    del texts, image_inputs
    # print(batch['input_ids'].shape)
    # print(batch['pixel_values'].shape)
    return batch


def main(script_args, training_args, model_args):
    torch.cuda.empty_cache()
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################

    dataset = load_dataset('json', data_files=script_args.dataset_name)

    ################
    # Load tokenizer
    ################
    min_pixels = 3136
    max_pixels = 200000
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, min_pixels=min_pixels, max_pixels=max_pixels
        )
        # if "Qwen2.5-VL" in model_args.model_name_or_path.lower():
        #     processor.image_processor.max_pixels = 4000
        #     # processor.image_processor.min_pixels = 3136
        #     logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    print(model_kwargs)
    # training_args.model_init_kwargs = model_kwargs
    # from transformers import Qwen2VLForConditionalGeneration
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     model_args.model_name_or_path, **model_kwargs
    # )
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        callbacks=[EmptyCacheCallback()]
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["R1-V"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
        processor.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
