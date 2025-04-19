# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import torch
from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False

class FixedNormalization1D(nn.Module):
    def __init__(self, channels):
        super(FixedNormalization1D, self).__init__()
        # Create a BatchNorm1d layer for b*h*w tensors
        self.bn = nn.BatchNorm1d(channels, affine=False)
        
        # # Convert std to variance and set running stats
        # mean_values = mean.squeeze()
        # var_values = std.squeeze()
        
        # # Initialize running stats
        # self.bn.running_mean = mean_values
        # self.bn.running_var = var_values**2
        
        # Set to eval mode to prevent stats from being updated
        self.bn.eval()
    
    def forward(self, x):
        # 保存原始维度
        # original_shape = x.shape
        # 确保输入是3维张量 [batch, 1, features]
        # x = x.view(original_shape[0], 1, -1)
        # 应用normalization
        x = self.bn(x)
        # 恢复原始维度
        # x = x.view(*original_shape)
        return x
        
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attentionv2(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dim = dim

        # 为每个head创建独立的qkv变换
        self.q_projs = nn.ModuleList([nn.Linear(dim, self.head_dim, bias=qkv_bias) for _ in range(num_heads)])
        self.k_projs = nn.ModuleList([nn.Linear(dim, self.head_dim, bias=qkv_bias) for _ in range(num_heads)])
        self.v_projs = nn.ModuleList([nn.Linear(dim, self.head_dim, bias=qkv_bias) for _ in range(num_heads)])
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
    @classmethod
    def from_pretrained(cls, attn_module: 'Attention', **kwargs):
        """从预训练的Attention模块加载参数到Attentionv2模块"""
        dim = attn_module.qkv.in_features
        num_heads = attn_module.num_heads
        head_dim = dim // num_heads
        
        # 创建新实例
        new_module = cls(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=attn_module.qkv.bias is not None,
            proj_bias=attn_module.proj.bias is not None,
            attn_drop=attn_module.attn_drop.p,
            proj_drop=attn_module.proj_drop.p,
            **kwargs
        )
        
        # 获取原qkv层的权重和偏置
        qkv_weight = attn_module.qkv.weight.data
        qkv_bias = attn_module.qkv.bias.data if attn_module.qkv.bias is not None else None
        
        # 分割qkv权重到各个head
        for head_idx in range(num_heads):
            # 分割q/k/v权重
            q_start = head_idx * head_dim
            q_end = (head_idx + 1) * head_dim
            k_start = dim + head_idx * head_dim
            k_end = dim + (head_idx + 1) * head_dim
            v_start = 2 * dim + head_idx * head_dim
            v_end = 2 * dim + (head_idx + 1) * head_dim
            
            # 设置q_proj权重
            new_module.q_projs[head_idx].weight.data = qkv_weight[q_start:q_end]
            # 设置k_proj权重
            new_module.k_projs[head_idx].weight.data = qkv_weight[k_start:k_end]
            # 设置v_proj权重
            new_module.v_projs[head_idx].weight.data = qkv_weight[v_start:v_end]
            
            if qkv_bias is not None:
                # 设置q_proj偏置
                new_module.q_projs[head_idx].bias.data = qkv_bias[q_start:q_end]
                # 设置k_proj偏置
                new_module.k_projs[head_idx].bias.data = qkv_bias[k_start:k_end]
                # 设置v_proj偏置
                new_module.v_projs[head_idx].bias.data = qkv_bias[v_start:v_end]
        
        # 复制proj层参数
        new_module.proj.weight.data = attn_module.proj.weight.data
        if attn_module.proj.bias is not None:
            new_module.proj.bias.data = attn_module.proj.bias.data
            
        return new_module

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        
        # 存储每个head的输出
        head_outputs = []
        
        # 对每个head单独计算attention
        for head_idx in range(self.num_heads):
            # 计算当前head的q, k, v
            q = self.q_projs[head_idx](x) * self.scale  # [B, N, head_dim]
            k = self.k_projs[head_idx](x)  # [B, N, head_dim]
            v = self.v_projs[head_idx](x)  # [B, N, head_dim]
            
            # 计算attention scores
            attn = (q @ k.transpose(-2, -1))  # [B, N, N]
            
            # 应用softmax
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            # 计算当前head的输出
            head_output = (attn @ v)  # [B, N, head_dim]
            head_outputs.append(head_output)
        
        # 将所有head的输出拼接
        x = torch.cat(head_outputs, dim=-1)  # [B, N, C]
        
        # 应用输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

        
class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class MemEffAttentionv2(Attentionv2):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        head_dim = self.head_dim
        num_heads = self.num_heads
        
        # 为每个head准备q, k, v
        q_list = []
        k_list = []
        v_list = []
        
        for head_idx in range(num_heads):
            # 计算当前head的q, k, v
            q = self.q_projs[head_idx](x) * self.scale  # [B, N, head_dim]
            k = self.k_projs[head_idx](x)  # [B, N, head_dim]
            v = self.v_projs[head_idx](x)  # [B, N, head_dim]
            
            # 添加到列表
            q_list.append(q.unsqueeze(1))  # [B, 1, N, head_dim]
            k_list.append(k.unsqueeze(1))  # [B, 1, N, head_dim]
            v_list.append(v.unsqueeze(1))  # [B, 1, N, head_dim]
        
        # 拼接所有head的q, k, v
        q = torch.cat(q_list, dim=1)  # [B, num_heads, N, head_dim]
        k = torch.cat(k_list, dim=1)  # [B, num_heads, N, head_dim]
        v = torch.cat(v_list, dim=1)  # [B, num_heads, N, head_dim]
        
        # 使用xformers的memory_efficient_attention
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        
        # 应用输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    @classmethod
    def from_mem_eff_attention(cls, mem_eff_attn: 'MemEffAttention', **kwargs):
        """从MemEffAttention模块创建MemEffAttentionv2模块"""
        # 首先创建一个Attentionv2实例
        attn_v2 = Attentionv2.from_pretrained(mem_eff_attn, **kwargs)
        
        # 创建MemEffAttentionv2实例
        new_module = cls(
            dim=mem_eff_attn.qkv.in_features,
            num_heads=mem_eff_attn.num_heads,
            qkv_bias=mem_eff_attn.qkv.bias is not None,
            proj_bias=mem_eff_attn.proj.bias is not None,
            attn_drop=mem_eff_attn.attn_drop.p,
            proj_drop=mem_eff_attn.proj_drop.p,
            **kwargs
        )
        
        # 复制Attentionv2的参数
        new_module.q_projs = attn_v2.q_projs
        new_module.k_projs = attn_v2.k_projs
        new_module.v_projs = attn_v2.v_projs
        new_module.proj = attn_v2.proj
        
        return new_module


def replace_attention_with_attentionv2(model):
    """将模型中的Attention替换为Attentionv2"""
    for name, module in model.named_children():
        if isinstance(module, Attention) and not isinstance(module, MemEffAttention):
            # 替换为Attentionv2
            setattr(model, name, Attentionv2.from_pretrained(module))
        elif isinstance(module, MemEffAttention):
            # 替换为MemEffAttentionv2
            setattr(model, name, MemEffAttentionv2.from_mem_eff_attention(module))
        else:
            # 递归处理子模块
            replace_attention_with_attentionv2(module)
    
    return model

        