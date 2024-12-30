# -*- encoding: utf-8 -*-
'''
Adapted from autoregressive_sampling.py in sat.
'''

import sys, os
sys.path.append(os.path.dirname(__file__))
import re
from functools import partial
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
import requests
from PIL import Image
from io import BytesIO
import torch
from sat.generation.autoregressive_sampling import BaseStrategy, get_masks_and_position_ids_default
from .com_utils import HARD_PROMPT
from tool_server.utils.cogcom.models.com_memory import update_mems




def filling_sequence(
        model, 
        seq, 
        batch_size,
        strategy=BaseStrategy(),
        max_memory_length=100000,
        log_attention_weights=None,
        get_masks_and_position_ids=get_masks_and_position_ids_default,
        turns_mems=None,
        turns_mems_mask=None,
        tokenizer=None,
        mem_lens=0,
        **kw_args
        ):
    '''
        seq: [2, 3, 5, ..., -1(to be generated), -1, ...]
        mems: [num_layers, batch_size, len_mems(index), mem_hidden_size]
            cache, should be first mems.shape[1] parts of context_tokens.
            mems are the first-level citizens here, but we don't assume what is memorized.
            input mems are used when multi-phase generation.
    '''
    assert len(seq.shape) == 1
    if hasattr(strategy, 'num_beams') and batch_size < strategy.num_beams:
        batch_size = strategy.num_beams
        print(f'Adjust batch_size to {batch_size} due to num_beams. Mute this warning by setting batch_size == num_beams.', level='DEBUG')

    # building the initial tokens, attention_mask, and position_ids
    context_length = 0
    while seq[context_length] >= 0:
        context_length += 1 # [0, context_length-1] are given
    assert context_length > 0
    tokens, attention_mask, position_ids = get_masks_and_position_ids(seq)
    tokens = tokens[..., :context_length]
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.type_as(next(model.parameters())) # if fp16
    # initialize generation
    counter = context_length - 1 # Last fixed index is ``counter'' 
    # index = 0 if mems is None else mems.shape[2] # Next forward starting index, also the length of cache.
    index = 0 # Next forward starting index, always setting to 0
    mems_cross = None
    # step-by-step generation
    mems, mems_mask = turns_mems, turns_mems_mask
    mem_lens = kw_args.get('mem_len')
    while counter < len(seq) - 1:
        # Now, we want to generate seq[counter + 1],
        # token[:, index: counter+1] needs forwarding.
        if mems_mask is not None:
            kw_args['mems_mask'][:, :mems_mask.shape[1]] = mems_mask

        if seq[counter + 1] >= 0: # provided
            tokens = torch.cat(
                (
                tokens, 
                    seq[counter+1: counter+2].expand(tokens.shape[0], 1)
                ), dim=1
            )
            counter += 1
            continue

        # forward
        if log_attention_weights is not None:
            log_attention_weights_part = log_attention_weights[..., index: counter+1, :counter+1] # TODO memlen
        else:
            log_attention_weights_part = None

        logits, *output_per_layers = model(
            input_ids=tokens[:, index:], 
            position_ids=position_ids[..., index: counter+1],
            # attention_mask=attention_mask[..., index: counter+1, :counter+1], # TODO memlen
            attention_mask=attention_mask[..., index: counter+1, index:counter+1], # TODO memlen
            mems=mems,
            mems_cross=mems_cross,
            log_attention_weights=log_attention_weights_part,
            **kw_args
        )
        if len(output_per_layers) > 0 and 'mem_cross' in output_per_layers[0]:
            mems_cross = [mem['mem_cross'] for mem in output_per_layers]
        mem_kv = [o['mem_kv'] for o in output_per_layers]
        # mems = update_mems(mem_kv, mems, max_memory_length=max_memory_length)
        mems, mems_mask = update_mems(mem_kv, mems, mems_mask, mem_lens, max_memory_length=max_memory_length) # [num_layers, batch, memory_length, 2d]
        mem_lens = torch.ones([1], device=mems.device)

        counter += 1
        index = counter
        # sampling
        logits = logits[:, -1].expand(batch_size, -1) # [batch size, vocab size]
        tokens = tokens.expand(batch_size, -1)
        tokens, mems = strategy.forward(logits, tokens, mems)
        if strategy.is_done:
            break
    return strategy.finalize(tokens, mems), mems_mask





def process_image(text, text_processor, img_processor, cross_img_processor, image=None):
    '''Process image in text.
    Args:
        text: str, text.
        image: Optional, image path / url / PIL image.
    '''
    image_position = text.rfind(text_processor.tokenizer.boi) + 5
    if image_position < 5:
        return text, image_position, (None, None, None)
    # extract path from [IMG][/IMG] using re
    pattern = (text_processor.tokenizer.boi + r"(.*?)" + text_processor.tokenizer.eoi).replace('[', r'\[').replace(']', r'\]')
    image_path = re.findall(pattern, text)
    image_path = image_path[-1] if image_path[-1] else None
    if image is None:
        assert image_path is not None, "image and image_path cannot be both None."
        text = text.replace(image_path, "")
        image_path = image_path.strip()
        # url
        if image_path.startswith("http"):
            response = requests.get(image_path, timeout=10)
            image = Image.open(BytesIO(response.content))
        # local path
        else:
            image = Image.open(image_path)
    if image is not None and isinstance(image, Image.Image):
        pil_img = image.convert('RGB')
        image = img_processor(pil_img) if img_processor is not None else {}
        cross_image = cross_img_processor(pil_img) if cross_img_processor is not None else {}
        # image = image.unsqueeze(0)
        ret = (image, pil_img, cross_image)
        if image_path:
            text = text.replace(image_path, "") # also remove the image path
    else:
        ret = image
    return text, image_position, ret


from .com_dataset import start_prompts
def chat(image_path, model, text_processor, img_processor, cross_img_processor,
        query: str, history: List[Tuple[str, str]] = None, image: Image = None,
        max_length: int = 1024, top_p=0.7, top_k=30, temperature=0.95, repetition_penalty=1.2,
        invalid_slices=[], no_prompt=False, add_preprompt=False, parse_result=False
        ):
    if add_preprompt:
        query = start_prompts[0].format(QUESTION=query)
    init_query = query
    turns_mems, turns_mems_mask = None, None
    ret_response, ret_history, ret_imgs = [],  [], []
    turns = 0
    while True: # multi-turn
        if turns > 0:
            query = HARD_PROMPT
        is_image_mode = image_path or (type(image) is not tuple and image is not None) or (type(image) is tuple and image != (None, None, None))
        if not history:
            history = []
        if is_image_mode:
            prompt = "{}{}{}".format(text_processor.tokenizer.boi, image_path if image_path else "", text_processor.tokenizer.eoi)
        else:
            prompt = ""
        if not is_image_mode or not no_prompt:
            # prompt += text_processor.history_to_prompt(history, query)
            prompt += text_processor.history_to_prompt(history=None, query=query)
        prompt, image_position, (torch_image, pil_img, cross_image) = process_image(prompt, text_processor, img_processor, cross_img_processor, image=image)
        if torch_image is not None:
            assert type(torch_image) is dict and type(cross_image) is dict
            if type(torch_image) is dict:
                for k in torch_image:
                    if type(torch_image[k]) is torch.Tensor and torch_image[k].dtype is not torch.int and torch_image[k].dtype is not torch.long:
                        torch_image[k] = torch_image[k].to(next(model.parameters()).dtype)
                    if type(torch_image[k]) is torch.Tensor:
                        torch_image[k] = torch_image[k].to(next(model.parameters()).device)
            else:
                torch_image = torch_image.to(next(model.parameters()).dtype).to(next(model.parameters()).device)
            
            for k in cross_image:
                if type(cross_image[k]) is torch.Tensor and cross_image[k].dtype is not torch.int and cross_image[k].dtype is not torch.long:
                    cross_image[k] = cross_image[k].to(next(model.parameters()).dtype)
                if type(cross_image[k]) is torch.Tensor:
                    cross_image[k] = cross_image[k].to(next(model.parameters()).device)
        
        if image_position < 5: # no image
            inputs = text_processor.tokenizer([prompt], return_tensors="pt").to(model.parameters().__next__().device)['input_ids'][0]
            # pre_image = 0
        else:
            new_prompt = prompt[image_position:]
            if not torch_image or hasattr(text_processor, 'no_eoi'):
                new_prompt = new_prompt.replace(text_processor.tokenizer.eoi, '', 1)
            inputs_dic = text_processor(new_prompt)
            for k in inputs_dic:
                if type(inputs_dic[k]) is torch.Tensor and inputs_dic[k].dtype is not torch.int and inputs_dic[k].dtype is not torch.long:
                    inputs_dic[k] = inputs_dic[k].to(next(model.parameters()).dtype)
                if type(inputs_dic[k]) is torch.Tensor:
                    inputs_dic[k] = inputs_dic[k].to(next(model.parameters()).device)
            inputs = inputs_dic['input_ids'].to(model.parameters().__next__().device)[0]
            # pre_image = inputs_dic['pre_image']
        
        seq = torch.cat(
            [inputs, torch.tensor([-1]*(max_length-len(inputs)), device=inputs.device)], dim=0
        )
        strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[text_processor.tokenizer.eos_token_id],
                                invalid_slices=invalid_slices, repetition_penalty=repetition_penalty)
        get_func = text_processor.get_func(inputs, **inputs_dic) if hasattr(text_processor, 'get_func') else get_masks_and_position_ids_default
        if image_position < 5:
            inputs = {}
        else:
            inputs = {**{'vision_'+k:v for k,v in torch_image.items()}, **{'cross_'+k:v for k,v in cross_image.items()}}
            inputs_dic.pop('input_ids')
            inputs = {**inputs, **inputs_dic}
            # inputs = {'vision_image': torch_image} if type(torch_image) is not dict else {'vision_'+k:v for k,v in torch_image.items()}
        # try:
        
        (output, turns_mems), turns_mems_mask = filling_sequence(
            model, seq,
            batch_size=1,
            get_masks_and_position_ids=get_func,
            strategy=strategy,
            turns_mems=turns_mems,
            turns_mems_mask=turns_mems_mask,
            tokenizer = text_processor.tokenizer,
            # pre_image=pre_image,
            **inputs
        )
        # except Exception as e:
        #     from sat.helpers import print_rank0
        #     print(f"test:{turns}, query:{query}, prompt:{prompt}, new_prompt:{new_prompt}, image_position:{image_position}, inputs_keys:{inputs.keys()}", flush=True)
        #     print(e, flush=True)
        #     assert 1==2
        
        # ---------------
        # port from inference_glm.py, more general than chat mode
        # clip -1s and fill back generated things into seq
        if type(output) is not list:
            output_list = output.tolist()
        else:
            output_list = output

        response = text_processor.tokenizer.decode(output_list[0])
        response = response.split(text_processor.sep)[-1].strip()
        # if hasattr(text_processor, 'process_response'):
        #     response = text_processor.process_response(response)
        # response = response.split(text_processor.sep)[-1].strip()
        # history = history + [(query, response)]

        drawn_img = None
        if parse_result:
            from grounding_parser import parse_response
            drawn_img = parse_response(pil_img, response, img_processor if img_processor is not None else cross_img_processor, output_name=f"output_turn{turns}.png")

        response, query, image, done = text_processor.process_response(response, pil_img) # handle manipulations
        ret_response.append(response)
        ret_history.append(response)
        ret_imgs.append((torch_image, pil_img, cross_image, drawn_img))
        if done or turns>=10:
            break
        turns += 1

    ret_response = ' '.join(ret_response)
    ret_response = ret_response[len(init_query):].strip()
    # return ret_response, ret_history, ret_imgs
    return ret_response, ret_history, ret_imgs[-1]

