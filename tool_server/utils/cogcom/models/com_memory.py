import os
import sys
import math
import random
import torch
import einops

from sat.model.base_model import BaseModel, BaseMixin, non_conflict
from sat.transformer_defaults import attention_fn_default

class CachedAutoregressiveCOMMixin(BaseMixin):
    def __init__(self):
        super().__init__()     
           
    @non_conflict
    def attention_fn(self, q, k, v, mask, dropout_fn, mems=None, cross_attention=False, old_impl=attention_fn_default,
                     **kw_args):
        
        if not cross_attention:
            mem = mems[kw_args['layer_id']] if mems is not None else None # 2, batch, head, seqlen, hidden_size
            b, nh, seq_len, hidden_size = k.shape

            cache_kv = torch.stack((k, v)).permute(1, 3, 0, 2, 4).detach().contiguous().view(b, seq_len, nh * hidden_size * 2)
            kw_args['output_this_layer']['mem_kv'] = cache_kv

            if mem is not None: # the first time, mem is None
                # might change batch_size
                mem = mem.expand(b, -1, -1).reshape(b, mem.shape[1], 2, nh, hidden_size).permute(2, 0, 3, 1, 4)
                memk, memv = mem[0], mem[1]
                k = torch.cat((memk, k), dim=2)
                v = torch.cat((memv, v), dim=2)
        else:
            kw_args['output_this_layer']['mem_cross'] = kw_args["encoder_outputs"]
            if q.shape[0] != k.shape[0]:
                k = k.expand(q.shape[0], *[-1]*(len(k.shape)-1))
            if q.shape[0] != v.shape[0]:
                v = v.expand(q.shape[0], *[-1]*(len(v.shape)-1))
        if mem is not None:
            mems_mask = kw_args['mems_mask']
            mems_mask = mems_mask[:, :mems[kw_args['layer_id']].shape[1]].unsqueeze(1).unsqueeze(1).repeat(1,1,mask.shape[2],1)
            # using non-tril attention mask
            mask = torch.cat([mems_mask, mask], dim=3)
            # Attach non-causal part of KV
            with torch.no_grad():
                mask_kvpart = torch.zeros(b, 1, mask.shape[2], seq_len-q.shape[2], device=mask.device)
                mask = torch.cat([mask, mask_kvpart], dim=3)
            mask = mask.bfloat16()

        return old_impl(q, k, v, mask, dropout_fn, cross_attention=cross_attention, mems=mems, **kw_args)



def update_mems(hiddens, mems, mems_mask, mem_lens, max_memory_length):
    '''
        hiddens: list (num_layers) of [batch, query_length, 2d]
        mems: None or [num_layers, batch, memory_length, 2d]
        mems_mask: [batch, memory_length],  1/0 means valid/not
        mem_lens: [batch, ],  current lengths of memories
    '''
    if hiddens is None:
        return None
    hiddens = torch.stack(hiddens)
    memory_length = mems.shape[2] if mems is not None else 0
    query_length = hiddens.shape[2]

    ls, b, s, d2 = hiddens.shape
    with torch.no_grad():
        cur_mems_mask = torch.zeros([b, s])
        cur_mems_mask = torch.where(torch.ones_like(cur_mems_mask, device=mem_lens.device).cumsum(dim=1)<=einops.repeat(mem_lens, 'b -> b s', s=s), 1, 0.)

    new_memory_length = min(max_memory_length, memory_length + query_length)
    with torch.no_grad():
        if new_memory_length <= query_length:
            # return hiddens[:, :, -new_memory_length:]
            mems= hiddens[:, :, -new_memory_length:]
            return mems, cur_mems_mask
        else:
            if mems.shape[1] < hiddens.shape[1]:
                mems = mems.expand(-1, hiddens.shape[1], -1, -1)
            mems= torch.cat(
                (mems[:, :, -new_memory_length+query_length:], hiddens),
                dim=2
            )
            mems_mask = torch.cat([mems_mask, cur_mems_mask], dim=1)
            mems_mask = mems_mask[:, -new_memory_length:]
        return mems, mems_mask