import abc
import hashlib
import json
import os
from typing import List, Optional, Tuple, Type, TypeVar, Union

from loguru import logger as eval_logger
from tqdm import tqdm
from gradio.helpers import Examples
import argparse
import base64
from collections import defaultdict
import copy
import datetime
from functools import partial
import json
import os
import torch
from pathlib import Path
import cv2
import numpy as np
import re
import time
from io import BytesIO
from PIL import Image
from PIL import Image as _Image  # using _ to minimize namespace pollution

import gradio as gr
from gradio import processing_utils, utils
from gradio_client import utils as client_utils

import requests

from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem

import pycocotools.mask as mask_util
import uuid

from tool_server.inferencer.utils.log_utils import build_logger

inferencer_id = str(uuid.uuid4())[:6]
logger = build_logger("abstract_model", f"abstract_model_{inferencer_id}.log")

R = partial(round, ndigits=2)
T = TypeVar("T", bound="tp_model")


class tp_model(abc.ABC):
    def __init__(
        self,
        inference_args,
        data_args,
    ):
        self.inference_args = inference_args
        self.data_args = data_args
        
        self.controller_addr = inference_args.controller_addr
        self.available_models = self.get_model_list()
        self.headers = {"User-Agent": "LLaVA-Plus Client"}
        self.init_model_addr_dict()
    
    def to(self, *args, **kwargs):
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
    
