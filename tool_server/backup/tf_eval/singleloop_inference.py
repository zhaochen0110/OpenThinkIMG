import os
import json
import tqdm
import torch
import re
import argparse
import datetime
import requests
from PIL import Image
from io import BytesIO
from peft import PeftModel
from copy import deepcopy
from torch.utils.data import DataLoader,Dataset
from dataclasses import dataclass, field

from typing import Dict, Sequence, Optional,List
from accelerate import PartialState,Accelerator
from tqdm import tqdm
from functools import partial
import threading
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer

from tool_server.tf_eval.inferencer import inferencer_dict

if __name__ == "__main__":

    @dataclass
    class DataArguments:

        input_path: str = field(default=None)
        output_path: str = field(default=None)
        image_dir_path: str = field(default=None)
        dataset_name: str = field(default="chartqa")
        batch_size: int = field(default=32)
        num_workers: int = field(default=4)
              
    @dataclass
    class InferenceArguments:
        model_name: str = field(default="Qwen2-VL-7B-Instruct")
        controller_addr: str = field(default="http://localhost:20001")
        inferencer_name: str = field(default="qwen2vl")
        max_rounds: int = field(default=3)

            
    parser = HfArgumentParser(
    (InferenceArguments, DataArguments))
    
    
    inference_args, data_args = parser.parse_args_into_dataclasses()
    inferencer_name = inference_args.inferencer_name
    
    inference_module = inferencer_dict[inferencer_name](inference_args=inference_args,data_args=data_args)
    inference_module.single_loop_inference()