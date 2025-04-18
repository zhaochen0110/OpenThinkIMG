
from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os
from datasets import Dataset

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

def load_dataset(file_path, num_samples=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    if num_samples is None:
        return Dataset.from_dict({"data": dataset})
    # indices = [0, 1, 3, 17, 23, 33]
    # indices = [1, 23]
    # extracted_list = [dataset[i] for i in indices]
    # return Dataset.from_dict({"data": extracted_list})
    # return Dataset.from_dict({"data": dataset[:num_samples]})
    # return Dataset.from_dict({"data": dataset[14121:14122]})
    return Dataset.from_dict({"data": dataset[:num_samples]})

def load_data_function():
    
    # raw_data = load_dir_of_jsonl_data_function_default(task_config)
    dataset_path = task_config["dataset_path"]
    image_dir_path = task_config["image_dir_path"]
    num_samples = task_config["num_sample"] 

    dataset = load_dataset(dataset_path, num_samples)

    meta_data = []
    for idx,item in enumerate(dataset):
        item = item["data"]
        item_id = f"chartqa_{idx}"
        image_file = item.get("image_file")

        image_path = os.path.join(image_dir_path, image_file)
        text = item["query"]
        data_item = dict(idx=item_id, text=text, **item)
        data_item["image_path"] = image_path
        meta_data.append(data_item)

    ## Show statistics
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data


def evaluate_function(results,meta_data):
    return {"results":results, "meta_data":meta_data}