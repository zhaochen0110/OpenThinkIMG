import collections
import inspect
import logging
import os,sys
from functools import partial
from typing import Dict, List, Mapping, Optional, Union

from loguru import logger


logger.remove()
logger.add(sys.stdout, level="WARNING")

# AVAILABLE_MODELS = {
#     "reasoneval": "ReasonEval",
#     "math_shepherd": "MathShepherd",
# }

def get_task_object(task_name,object_name):
    try:
        module = __import__(f"mr_eval.tasks.{task_name}.task", fromlist=[object_name])
        return getattr(module, object_name)
    except Exception as e:
        logger.error(f"Failed to import {object_name} from {task_name}: {e}")
        raise
    

def get_task_functions(task_name):
    '''
    return a dictionary of functions from the task module
    {
        "load_data_function": load_data_function,
        "evaluate_function": evaluate_function,
        "task_config": task_config
    }
    '''
    function_list = ["load_data_function","evaluate_function","task_config"]
    try:
        module = __import__(f"mr_eval.tasks.{task_name}.task", fromlist=["*"])
        res_dict = {func:getattr(module, func) for func in function_list}
        return res_dict
    
    except Exception as e:
        logger.error(f"Failed to import all functions from {task_name}: {e}")
        raise
