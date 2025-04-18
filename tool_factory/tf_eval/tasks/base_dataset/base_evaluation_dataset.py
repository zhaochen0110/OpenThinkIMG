import torch
import argparse
import datetime
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoTokenizer
from copy import deepcopy
from torch.utils.data import DataLoader,Dataset
from dataclasses import dataclass
from typing import Dict, Sequence
from accelerate import PartialState  
from tqdm import tqdm
import os
import torch.distributed as dist

from ...utils.utils import gather_dict_lists, append_jsonl, process_jsonl, is_vllm_environment
from ...utils.log_utils import get_logger

logger = get_logger(__name__)
class BaseEvalDataset(Dataset):
    '''
    The API dataset class
    Initialize: load_data_function is provided by task
                evaluate_function is provided by task
                getitem_function is provided by model
    data storage: self.meta_data
    results storage: self.results
    '''
    def __init__(
        self,
        load_data_function,
        getitem_function,
        evaluate_function, 
        task_config=None,
        task_args=None,
        model_args=None,
    ) -> None:
        self.load_data_function = load_data_function
        self.getitem_function = getitem_function
        self.evaluate_function = evaluate_function
        self.task_config = task_config
        self.task_args = task_args
        self.model_args = model_args
        self.task_name = task_config["task_name"]  
        self.model_name = model_args.model 
        
        if self.task_config and "generation_config" in self.task_config:
            self.set_gen_kwargs(self.task_config["generation_config"])
        else:
            self.set_gen_kwargs({})
        
        self.results = []
        
        self.full_data = self.load_data_function()
        self.meta_data = deepcopy(self.full_data)
        self.load_ckpt_path = None
        self.save_ckpt_path = None
        
        if task_args.resume_from_ckpt and self.task_name in self.task_args.resume_from_ckpt:
            self.load_ckpt_path = self.task_args.resume_from_ckpt[self.task_name]
            self.resume_from_ckpt(self.load_ckpt_path)
        
        if task_args.save_to_ckpt and self.task_name in self.task_args.save_to_ckpt:
            logger.info(f"save to ckpt path: {self.task_args.save_to_ckpt}")
            self.save_ckpt_path = self.task_args.save_to_ckpt[self.task_name]
        
        if self.model_name in ["qwen_qwq"]:
            logger.info("Generation model detected, setting padding side to left")
            self.padding_side = "left"
        else:
            self.padding_side = "right"
            
        if dist.is_available() and dist.is_initialized() and not 'vllm' in self.model_name:
            dist.barrier()
            
    
    def __getitem__(self, index):
        return self.getitem_function(self.meta_data,index)
    
    def __len__(self):
        return len(self.meta_data)
    
    def store_results(self,result):
        self.results.append(result)
        if self.save_ckpt_path:
            self.save_item_into_ckpt_file(result)
        
    def fetch_results(self):
        return self.results
    
    def evaluate(self):
        self.collect_results_from_multi_process()
        res = self.evaluate_function(results=self.results, meta_data=self.full_data)
        res["task_name"] = self.task_name
        res["model_name"] = self.model_name
        return res
    
    def save_result_item_into_log(self,result_item, save_path):
        append_jsonl(result_item, save_path)
        
    def resume_from_ckpt(self,ckpt_path):
        if os.path.exists(ckpt_path):
            logger.info(f"loading results from {ckpt_path}")
            ckpt_data = process_jsonl(ckpt_path)
            self.processed_id = {}
            
            for ckpt_item in ckpt_data:
                # assert ckpt_item["task_name"] == self.task_name, f"ckpt task name {ckpt_item['task']} not match with current task name {self.task_name}"
                assert ckpt_item["model_name"] == self.model_name, f"ckpt model name {ckpt_item['model_name']} not match with current model name {self.model_name}"
                if "results" in ckpt_item and isinstance(ckpt_item["results"],dict): 
                    # and ("validity" not in ckpt_item["results"] or ckpt_item["results"]["validity"] == True):
                    self.results.append(ckpt_item["results"])
                    self.processed_id[ckpt_item["results"]["idx"]] = 1
                    
            self.meta_data = [item for item in self.meta_data if item["idx"] not in self.processed_id]
            logger.info(f"Total items: {len(self.full_data)}, processed items: {len(self.results)}, remaining items: {len(self.meta_data)}")
        else:
            logger.info(f"ckpt path {ckpt_path} not found")

    
    def save_item_into_ckpt_file(self,result_item):
        write_item = dict(task_name=self.task_name,model_name=self.model_name,results=result_item)
        append_jsonl(write_item, self.save_ckpt_path)
        
        
    def collect_results_from_multi_process(self):
        # breakpoint()
        if dist.is_available() and dist.is_initialized() and not 'vllm' in self.model_name:
            dist.barrier()
        self.results = gather_dict_lists(self.results)
        results_dict = {}
        renewed_results = []
        for item in self.results:
            if item["idx"] not in results_dict:
                results_dict[item["idx"]] = 1
                renewed_results.append(item)
        self.results = renewed_results
    
    def set_gen_kwargs(self, config):
        if isinstance(config, dict):
            self.gen_kwargs = config
        else:
            self.gen_kwargs = {}



@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for Evaluation."""
    tokenizer: AutoTokenizer
    max_length: int = 512
    padding_side: str = "left"

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        if "model_type" in instances[0] and instances[0]["model_type"] == "openai":
            assert len(instances) == 1
            return instances[0]
        
        idx = [instance["idx"] for instance in instances]
        input_ids = [instance["input_ids"] for instance in instances]
        
        assert isinstance(input_ids[0], torch.Tensor) 
        for input_id in input_ids:
            while input_id.ndim > 1:
                input_id = input_id[0]
                
        lengths  = [input_id.shape[0] for input_id in input_ids]
        max_length = max(lengths)
        max_length = min(max_length, self.max_length)
        pad_token_id = self.tokenizer.pad_token_id
        
        padded_batch = torch.zeros((len(input_ids), max_length), dtype=torch.long)
        attention_mask = torch.zeros((len(input_ids), max_length), dtype=torch.long)

        # 填充张量并生成注意力掩码
        for i, input_id in enumerate(input_ids):
            assert isinstance(input_id, torch.Tensor)
            if input_id.shape[0] > max_length:
                input_id = input_id[:max_length]
            length = min(input_id.shape[0], max_length)
            if self.padding_side == "right":
                padded_batch[i, :length] = input_id
                if length < max_length:
                    padded_batch[i, length:] = pad_token_id
                attention_mask[i, :length] = 1
            elif self.padding_side == "left":
                padded_batch[i, -length:] = input_id
                if length < max_length:
                    padded_batch[i, :-length] = pad_token_id
                attention_mask[i, -length:] = 1

        batch = dict(
            idx=idx,
            input_ids=padded_batch,
            attention_mask=attention_mask,
        )
        
        instance0 = instances[0]
        other_keys = [key for key in instance0.keys() if key not in ["idx","input_ids"]]
        if len(other_keys) > 0:
            for key in other_keys:
                values = [instance[key] for instance in instances]
                batch[key] = values
        return batch

