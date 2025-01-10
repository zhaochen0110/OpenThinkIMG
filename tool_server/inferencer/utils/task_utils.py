from .utils import *
from box import Box
        
def load_task_config(file_path):
    task_config = load_yaml_file(file_path)
    task_config = Box(task_config)
    return task_config

def load_data_function_default(task_config):
    return load_jsonl_data_function_default(task_config)

def load_jsonl_data_function_default(task_config):
    task_name = task_config["task_name"]
    dataset_type = task_config["dataset_type"]
    if dataset_type == "jsonl":
        dataset_path = task_config["dataset_path"]
        meta_data = process_jsonl(dataset_path)
    elif dataset_type == "json":
        dataset_path = task_config["dataset_path"]
        meta_data = load_json_file(dataset_path)
    else:
        raise ValueError(f"dataset_type {dataset_type} not supported")
    return meta_data

def load_dir_of_jsonl_data_function_default(task_config):
    task_name = task_config["task_name"]
    dataset_type = task_config["dataset_type"]
    dataset_path = task_config["dataset_path"]
    assert dataset_type == "dir_of_jsonl"
    assert os.path.isdir(dataset_path)
    files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".jsonl")]
    meta_data = []
    for file in files:
        meta_data.extend(process_jsonl(file))
    return meta_data


def get_task_config_from_current_dir(task_file_path):
    current_dir = os.path.dirname(os.path.abspath(task_file_path))
    file_name = "config.yaml"
    config_path = os.path.join(current_dir, file_name)
    task_config = load_task_config(config_path)
    if "dataset_path" in task_config and os.path.isabs(task_config["dataset_path"]) == False:
        task_config["dataset_path"] = os.path.join(current_dir,task_config["dataset_path"])
    return task_config
    
    
    