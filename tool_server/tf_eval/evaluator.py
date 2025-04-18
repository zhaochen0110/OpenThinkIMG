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
# from .utils.evaluate import evaluate_metric
from .tool_inferencer import BaseToolInferencer
import pdb
import re
from math_verify import parse, verify

logger = get_logger(__name__)

class TFEvaluator():
    def __init__(self, model_args, task_args, script_args):
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
            model_mode=self.model_args.model_mode,
            max_rounds = max_rounds,
            stop_token = stop_token,
            controller_addr = self.script_args.controller_addr,
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
            # pdb.set_trace()
            self.inferencer.batch_inference(dataset)
            # breakpoint()
            res_log = dataset.evaluate()
            # if is_main_process():
            #     logger.info(f"evaluation of {task_name} completed")
            #     append_jsonl(res_log, self.script_args.output_path)
            
            # pdb.set_trace()
            result_data = []
            with open(self.script_args.output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    result_data.append(json.loads(line))
                
            if 'reachqa' in self.script_args.output_path:
                ground_truth_path = '/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/test_dataset/reachqa200.json'
            elif 'chartgemma' in self.script_args.output_path:
                ground_truth_path = '/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/test_dataset/chartgemma200.json'
            
            ground_truth = dict()
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            for data in all_data:
                ground_truth[data['question']] = data['label'].replace('<answer> ', '').replace(' </answer>', '')

            processed_data = dict()
            error_cnt = 0
            pattern_list = [r'"actions": \[\{"name": "Terminate", "arguments": \{"ans": (.*?)\}', r'"actions": \[\{"name": "Terminate", "arguments": \{"answer": (.*?)\}']
            for data in result_data:
                # pdb.set_trace()
                try:
                    model_response = (data['results']['results']['conversation'][-1]['content'][0]['text'])
                    final_action = "{\"actions\": " + model_response.split("\"actions\": ")[1]
                    # print(final_action)
                    for pattern in pattern_list:
                        matches = re.findall(pattern, final_action)
                        if len(matches) == 1:
                            pred = matches[0]
                            pred = pred.replace(r'"', r'')
                            processed_data[data['results']['results']['meta_data']['text']] = pred
                    else:
                        raise Exception

                except Exception as e:
                    error_cnt += 1
                    
            acc = 0
            for item in processed_data.items():
                if item[0] not in ground_truth:
                    print('error')
                    continue
                gold = parse('${0}$'.format(ground_truth[item[0]]))
                pred = parse('${0}$'.format(item[1]))
                if verify(gold, pred):
                    acc+=1
                elif '%' in item[1][-1]:
                    pred = item[1][:-1:]
                    if '.' in pred:
                        pred = pred.split('.')[0]
                    if pred == ground_truth[item[0]]:
                        acc+=1

                elif '.' in item[1]:
                    pred = item[1].split('.')[0]
                    if pred == ground_truth[item[0]]:
                        acc+=1
                else:
                    pass
            print(acc/len(result_data))
            
