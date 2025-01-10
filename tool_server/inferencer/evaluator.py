import itertools
import json
import logging
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch, gc
import yaml
import argparse
from torch.utils.data import DataLoader
import torch.distributed as dist

from tool_server.inferencer.models import get_model
from tool_server.inferencer.tasks import get_task_object, get_task_functions
from tool_server.inferencer.tasks.base_dataset.base_evaluation_dataset import BaseEvalDataset, DataCollatorForSupervisedDataset

from tool_server.inferencer.utils.utils import *
from tool_server.inferencer.utils.arguments import *

from tool_server.inferencer.utils.log_utils import get_logger

logger = get_logger(__name__)

class TFEvaluator():
    def __init__(self,model_args, task_args, script_args):
        self.config = script_args.config
        self.model_args = model_args
        self.task_args = task_args
        self.script_args = script_args
        self.tasks = self.task_args.task_name
        self.model = get_model(self.model_args.model)(**self.model_args.model_args)
        self.tokenizer = self.model.tokenizer
        
    
    def evaluate(self):

        for task_name in self.tasks:
            logger.info(f"evaluating {task_name}")
            task_dict = get_task_functions(task_name)
            load_data_function, evaluate_function, task_config = task_dict["load_data_function"], task_dict["evaluate_function"], task_dict["task_config"]
            self.model.set_generation_config(task_config.generation_config)

            dataset = BaseEvalDataset(
                load_data_function=load_data_function,
                getitem_function=self.model.getitem_function,
                evaluate_function=evaluate_function,
                task_config = task_config,
                task_args = self.task_args,
                model_args = self.model_args,
            )
            data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer, max_length=task_config.generation_config.max_length, padding_side=dataset.padding_side)
            dataloader = DataLoader(dataset, batch_size=self.model_args.batch_size, num_workers=4, collate_fn=data_collator)
            self.model.respond(dataloader)
            res_log = dataset.evaluate()
            if is_main_process():
                logger.info(f"evaluation of {task_name} completed")
                append_jsonl(res_log, self.script_args.output_path)
                
                
            
            
