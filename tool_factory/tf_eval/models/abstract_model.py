import abc
from typing import List, Optional, Tuple, Type, TypeVar, Union


from functools import partial

from ...utils.utils import *
from ...utils.server_utils import *
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem

import uuid

from ..utils.log_utils import get_logger

inferencer_id = str(uuid.uuid4())[:6]
logger = get_logger("abstract_model",)

R = partial(round, ndigits=2)
T = TypeVar("T", bound="tp_model")


class tp_model(abc.ABC):
    def __init__(
        self,
    ):
        pass
    
    def to(self, *args, **kwargs):
        # import pdb; pdb.set_trace()
        self.model = self.model.to(*args, **kwargs)
        return self
    
    def eval(self):
        self.model = self.model.eval()
    
    def generate_conversation_fn(
        self,
        text,
        image, 
        role = "user",
    ):
        raise NotImplementedError
    
    def append_conversation_fn(
        self, 
        conversation, 
        text, 
        image, 
        role
    ):
        raise  NotImplementedError
    
    def generate(
        self,
        batch: List[DynamicBatchItem],
    ):
        raise NotImplementedError
    
    def getitem_fn(
        self,
        meta_data: List,
        idx: int,
    ):
        raise NotImplementedError
    
    def set_generation_config(self, generation_configs: dict = None) -> None:
        if generation_configs is not None:
            self.generation_config = generation_configs
        else:
            self.generation_config = {}