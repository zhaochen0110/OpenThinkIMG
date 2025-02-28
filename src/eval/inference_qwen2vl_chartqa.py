import argparse
from datasets import Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import os
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on the dataset.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input dataset file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--image_base_path", type=str, required=True, help="Base path for images.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output results.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process (default is all samples).")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Size of tensor parallelism (default is 4).")
    return parser.parse_args()

def load_dataset(file_path, num_samples=None):
    """Load a subset of the dataset from the given file path."""
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    if num_samples is None:
        return Dataset.from_dict({"data": dataset})
    return Dataset.from_dict({"data": dataset[:num_samples]})

def initialize_model(model_path, tensor_parallel_size, max_model_len=17000):
    """Initialize the LLaVA model."""
    return LLM(model=model_path, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size)

def _add_speaker_and_signal_no_answer(conversations):
    """Directly use the value from the conversation without adding speaker signals, excluding the GPT's answer."""
    conversation = ""
    for sentence in conversations:
        from_str = sentence["from"].lower()
        if from_str == "human":
            conversation += sentence["value"] + "\n"
        elif from_str == "gpt":
            continue
    return conversation

def preprocess_image(image_path, image_base_path):
    """Preprocess the image for model input."""
    full_image_path = os.path.join(image_base_path, image_path)
    image = Image.open(full_image_path).convert("RGB")
    return image

def run_inference_and_save(file_path, model_path, image_base_path, output_path, num_samples, tensor_parallel_size, batch_size=20000):
    """Run inference on the dataset and save the results."""
    
    dataset = load_dataset(file_path, num_samples)
    llm = initialize_model(model_path, tensor_parallel_size)

    total_samples = len(dataset)
    # import pdb; pdb.set_trace()
    generate_samples = []
    
    for i in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
        batch = dataset.select(range(i, min(i + batch_size, total_samples)))
        batch_inputs = []
        batch_data = []

        for sample in batch:
            sample = sample["data"]
            question = sample["query"]
            image_path = sample.get("imgname")
            if not image_path:
                continue
            

            # prompt = (f"<|im_start|>system<|im_end|>\n"
            #         "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            #         f"{question} First output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.\nResponse:\n<|im_end|>\n"
            #         "<|im_start|>assistant\n")
            prompt = (f"<|im_start|>system<|im_end|>\n"
                    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                    f"{question}\nResponse:\n<|im_end|>\n"
                    "<|im_start|>assistant\n")
            stop_token_ids = None


            # import pdb; pdb.set_trace()
            image = preprocess_image(image_path, image_base_path)
            
            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image
                },
            })
            batch_data.append(sample)
        # import pdb; pdb.set_trace()
        sampling_params = SamplingParams(temperature=0, max_tokens=2048, stop_token_ids=stop_token_ids)
        try:
            outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
        except ValueError as e:
            print(f"Error during model inference: {e}")
            continue
        
        for idx, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            image_path = batch_data[idx]["imgname"]
            label = batch_data[idx]["label"]
            query = batch_data[idx]["query"]
            generate_samples.append({
                "image_path": image_path,
                "question": query,
                "label": label,
                "generate_label": generated_text,
            })
    
    with open(output_path, "w") as f:
        json.dump(generate_samples, f, indent=4)
    
    print(f"Saved {len(generate_samples)} samples to {output_path}")

if __name__ == "__main__":
    args = parse_args()  # Parse arguments from command line
    run_inference_and_save(args.file_path, args.model_path, args.image_base_path, args.output_path, args.num_samples, args.tensor_parallel_size)









