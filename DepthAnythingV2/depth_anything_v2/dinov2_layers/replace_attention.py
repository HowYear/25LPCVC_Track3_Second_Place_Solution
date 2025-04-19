# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from .attention import Attention, MemEffAttention, Attentionv2, MemEffAttentionv2


def replace_attention_with_v2(model):
    """
    Replace all instances of Attention and MemEffAttention in the model with
    their v2 versions (Attentionv2 and MemEffAttentionv2).
    
    Args:
        model: The model to modify
        
    Returns:
        The modified model
    """
    for name, module in model.named_children():
        if isinstance(module, Attention) and not isinstance(module, MemEffAttention):
            # Replace with Attentionv2
            setattr(model, name, Attentionv2.from_pretrained(module))
        elif isinstance(module, MemEffAttention):
            # Replace with MemEffAttentionv2
            setattr(model, name, MemEffAttentionv2.from_mem_eff_attention(module))
        else:
            # Recursively process child modules
            replace_attention_with_v2(module)
    
    return model