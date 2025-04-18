
from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os
from datasets import Dataset

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

# def load_dataset(file_path, already_processed_path, num_samples=None):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         dataset = json.load(f)
#     process_data = set()
#     with open(already_processed_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line)
#             process_data.add(data['results']['results']['meta_data']['text'])
#     selected_dataset = []
#     for data in dataset:
#         if data['question'] not in process_data:
#             selected_dataset.append(data)
    
#     if num_samples is None:
#         return Dataset.from_dict({"data": selected_dataset})
#     return Dataset.from_dict({"data": selected_dataset[:num_samples]})

def load_dataset(file_path, num_samples=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    if num_samples is None:
        return Dataset.from_dict({"data": dataset})
    return Dataset.from_dict({"data": dataset[:num_samples]})

def load_data_function():
    
    dataset_path = task_config["dataset_path"]
    image_dir_path = task_config["image_dir_path"]
    num_samples = None

    dataset = load_dataset(dataset_path, num_samples)

    meta_data = []
    for idx,item in enumerate(dataset):
        item = item["data"]
        item_id = f"reachqa_{idx}"
        image_file = item.get("image_path")

        image_path = image_file
        text = item["question"]
        data_item = dict(idx=item_id, text=text, **item)
        data_item["image_path"] = image_path
        meta_data.append(data_item)

    ## Show statistics
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data


def evaluate_function(results,meta_data):
    return {"results":results, "meta_data":meta_data}