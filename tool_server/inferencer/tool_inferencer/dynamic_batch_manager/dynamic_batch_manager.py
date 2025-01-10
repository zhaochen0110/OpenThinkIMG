from dataclasses import dataclass
from typing import Dict, Sequence, Optional,List
from tool_server.inferencer.utils.log_utils import get_logger

logger = get_logger(__name__)

@dataclass
class DynamicBatchItem:

    max_rounds: int
    current_round : int
    status: str = "pending" # pending, processing, finished
    meta_data: Dict
    conversation: object = None
    model_response: str = ""
    tool_cfg = None
    tool_response = None
    new_round_input = None
    
    
    


class DynamicBatchManager():
    def __init__(
        self,
        batch_size: int,
        stop_token: str = "<stop>",
        max_rounds: int = 3,
        generate_conversation_fn = None,
    ):
        self.dynamic_batch = []
        self.batch_size = batch_size
        self.stop_token = stop_token
        self.max_rounds = max_rounds
        self.generate_conversation_fn = generate_conversation_fn
        
    
        
    def pop_qualified_items(self):
        res = []
        new_batch = []
        for idx,item in enumerate(self.dynamic_batch):
            if item.status == "finished":
                res.append(item.meta_data)
            else:
                new_batch.append(item)
        self.dynamic_batch = new_batch
        return res
    
    def append_item(self, meta_data: Dict):
        if len(self.dynamic_batch) < self.batch_size:
            candidate_item = DynamicBatchItem(
                max_rounds=self.max_rounds,
                current_round=0,
                meta_data=meta_data,
                status="pending"
            )
            candidate_item.conversation = self.generate_conversation_fn(
                text = meta_data["text"], 
                image = meta_data["image"],
                role = "user"
            )
            self.dynamic_batch.append(candidate_item)
        else:
            raise ValueError("Batch is full")
        
    def append_item_to_full(self,generator,progress_bar=None):
        for idx,item in enumerate(generator):
            try:
                self.append_item(item)
                if progress_bar is not None:
                    progress_bar.update(1)
            except Exception as e:
                logger.info(f"Batch is full, yielding {idx+1} items")
                return idx+1
    

    def get_current_batch(self):
        return self.dynamic_batch
    
    
    # Caution: Only model.generate can call this function
    def update_item_status(self):
        for item in self.dynamic_batch:
            if item.status == "pending":
                if item.current_round == item.max_rounds:
                    item.status = "finished"
                else:
                    item.current_round += 1
                    item.status = "processing"
            elif item.status == "processing":
                if item.current_round == item.max_rounds:
                    item.status = "finished"
            elif item.status == "finished":
                pass
            else:
                raise ValueError(f"Invalid status {item.status}")
        
    
    