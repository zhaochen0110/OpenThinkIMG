

from torch.utils.data import Dataset
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
from tool_server.tf_eval.prompts.qwen72b_assistant import assistant_prompt

logger = build_logger(__name__, f"{__name__}.log")


class BaseSingleLoopDataset(Dataset):
    def __init__(self, data_args):
        self.input_path = data_args.input_path
        self.output_path = data_args.output_path
        self.image_dir_path = data_args.image_dir_path
        self.resume_from_checkpoint()
        self.load_data()
    
    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        return self.meta_data[idx]
    
    

class ChartQASingleLoopDataset(BaseSingleLoopDataset):
    
    def resume_from_checkpoint(self):
        if os.path.exists(self.output_path):
            logger.info(f"Ckpt exists, loading from {self.output_path}")
            self.cache = process_jsonl(self.output_path)
            self.processed_id = {item["idx"] for item in self.cache}
        else:
            logger.info(f"Ckpt does not exist, creating new")
            self.cache = []
            self.processed_id = {}
    
    def load_data(self):
        raw_data_dict = load_json_file(self.input_path)
        raw_data = []
        for k,v in raw_data_dict.items():
            raw_data.append(v)
            
        self.meta_data = []
        for idx,item in enumerate(raw_data):
            figure_id = item["figure_id"]
            item_id = f"chartqa_{figure_id}_{idx}"
            if item_id in self.processed_id:
                continue
            image_path = os.path.join(self.image_dir_path, f"{figure_id}.jpg")
            prompt = assistant_prompt + '\n' + item["query"]
            data_item = dict(idx=item_id, image=image_path, prompt=prompt, gen_kwargs={}, **item)
            self.meta_data.append(data_item)
    
    def write_output_item(self, item):
        res=dict(
            idx=item["idx"],
            figure_id=item["figure_id"],
            query=item["query"],
            answer=item["answer"],
            inst_category=item["inst_category"],
            qa_source=item["qa_source"],
            value_list=item["value_list"],
            response_list=item["response_list"],
            tool_list=item["tool_list"],
        )
        append_jsonl(res, self.output_path)

            
