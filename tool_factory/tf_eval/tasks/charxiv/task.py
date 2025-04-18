
from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)


def load_data_function():
    
    # raw_data = load_dir_of_jsonl_data_function_default(task_config)
    dataset_path = task_config["dataset_path"]
    image_dir_path = task_config["image_dir_path"]
    num_samples = task_config["num_sample"]

    raw_data = []
    for k,v in load_json_file(dataset_path).items():
        raw_data.append(v)
        
    meta_data = []
    for idx,item in enumerate(raw_data):
        figure_id = item["figure_id"]
        item_id = f"charxiv_{figure_id}_{idx}"
        image_path = os.path.join(image_dir_path, f"{figure_id}.jpg")
        text = item["query"]

        data_item = dict(idx=item_id, image_path=image_path, text=text, **item)
        meta_data.append(data_item)
    meta_data = meta_data[:num_samples]
    ## Show statistics
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data


def evaluate_function(results,meta_data):
    return {"results":results,"meta_data":meta_data}