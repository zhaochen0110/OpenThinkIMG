import argparse
import json
import logging
import os
import sys
from functools import partial
from typing import Union
import gc
from accelerate import Accelerator

from tool_server.tf_eval.utils.utils import *
from tool_server.tf_eval.evaluator import TFEvaluator
from tool_server.tf_eval.utils.arguments import *
from tool_server.tf_eval.utils.log_utils import get_logger

logger = get_logger(__name__)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    accelerate = Accelerator()
    args_dict = parse_args()
    model_args, task_args, script_args = args_dict["model_args"], args_dict["task_args"], args_dict["script_args"]
    
    config = script_args.config
    if config:
        if not isinstance(config, list):
            config = [config]
            
        for idx, config_item in enumerate(config):
            dist_wait_for_everyone()
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"After Cleaning: Memory Allocated: {torch.cuda.memory_allocated()/(1024 ** 3) :.2f} GB")
            logger.info(f"Begin evaluating on the No. {idx+1} config, toal {len(config)} configs.")
            if isinstance(config_item, dict):
                model_args = ModelArguments(**config_item["model_args"])
                task_args = TaskArguments(**config_item["task_args"])
                script_args = ScriptArguments(**config_item["script_args"])
             
                task_args.task_name = parse_str_into_list(task_args.task_name)
                if isinstance(model_args.model_args, str):
                    model_args.model_args = parse_str_into_dict(model_args.model_args)
                if isinstance(script_args.wandb_args, str):
                    script_args.wandb_args = parse_str_into_dict(script_args.wandb_args)
            else:
                assert len(config) == 1, "If config is not a list, it should be a dictionary or NoneType"
                raise ValueError("Config should be a list of dictionaries.")
            
        evaluator = TFEvaluator(model_args, task_args, script_args)
        evaluator.evaluate()
        del evaluator
        logger.info(f"Finished evaluating on the No. {idx+1} config, toal {len(config)} configs.")
    else:
        evaluator = TFEvaluator(model_args, task_args, script_args)
        evaluator.evaluate()
        del evaluator
        logger.info("Finished evaluating on the single config.")

if __name__ == "__main__":
    main()