from dataclasses import dataclass, field, asdict
from typing import Dict, Sequence, Optional,List
from tool_server.tf_eval.utils.log_utils import get_logger
from ...utils.utils import *
from PIL import Image

logger = get_logger(__name__)

@dataclass
class DynamicBatchItem:
    max_rounds: int
    current_round : int
    status: str = "pending" # pending, processing, finished
    meta_data: Dict = field(default = None)
    conversation: object = field(default = None)
    model_response: List[str] = field(default_factory=list)
    tool_cfg :  List[str] = field(default_factory=list)
    tool_response :  List[str] = field(default_factory=list)
    new_round_input :  List[str] = field(default_factory=list)
    current_image : Image = field(default=None)
    
    


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
                item = asdict(item)
                item = remove_pil_objects(item)
                res.append(item)
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
    
    
    def append_item_to_full(self, dataloader, progress_bar=None):
        while len(self.dynamic_batch) < self.batch_size:
            try:
                # breakpoint()
                self.append_item(next(dataloader))
                if progress_bar:
                    progress_bar.update(1)
            except:
                break
        
    

    def get_current_batch(self):
        return self.dynamic_batch
    
    
    # Caution: Only model.generate can call this function
    def update_item_status(self):
        for item in self.dynamic_batch:
            if item.status == "pending":
                if item.current_round == item.max_rounds or "Terminate" in item.model_response[-1]:
                    item.status = "finished"
                else:
                    item.current_round += 1
                    item.status = "processing"
            elif item.status == "processing":
                if item.current_round == item.max_rounds or "Terminate" in item.model_response[-1]:
                    item.status = "finished"
                else:
                    item.current_round += 1
            elif item.status == "finished":
                pass
            else:
                raise ValueError(f"Invalid status {item.status}")
        
    
    