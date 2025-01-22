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

from .models import get_model
from .tasks import get_task_object, get_task_functions
from .tasks.base_dataset.base_evaluation_dataset import BaseEvalDataset, DataCollatorForSupervisedDataset

from .utils.utils import *
from .utils.arguments import *

from .utils.log_utils import get_logger, set_verbosity
from .tool_inferencer import BaseToolInferencer

logger = get_logger(__name__)

class TFEvaluator():
    def __init__(self,model_args, task_args, script_args):
        self.config = script_args.config
        self.model_args = model_args
        self.task_args = task_args
        self.script_args = script_args
        self.tasks = self.task_args.task_name
        self.model = get_model(self.model_args.model)(**self.model_args.model_args)
        max_rounds = self.model_args.max_rounds
        stop_token = self.model_args.stop_token
        
        set_verbosity(self.script_args.verbosity)
        
        self.inferencer = BaseToolInferencer(
            tp_model=self.model,
            batch_size=self.model_args.batch_size,
            max_rounds = max_rounds,
            stop_token = stop_token,
        )
    
    def evaluate(self):

        for task_name in self.tasks:
            logger.info(f"evaluating {task_name}")
            task_dict = get_task_functions(task_name)
            load_data_function, evaluate_function, task_config = task_dict["load_data_function"], task_dict["evaluate_function"], task_dict["task_config"]
            self.model.set_generation_config(task_config.generation_config)
            # Generate the first batch
            
            dataset = BaseEvalDataset(
                load_data_function=load_data_function,
                getitem_function=self.model.getitem_fn,
                evaluate_function=evaluate_function,
                task_config = task_config,
                task_args = self.task_args,
                model_args = self.model_args,
            )
            self.inferencer.batch_inference(dataset)
            res_log = dataset.evaluate()
            if is_main_process():
                logger.info(f"evaluation of {task_name} completed")
                append_jsonl(res_log, self.script_args.output_path)
                
                
            
            
