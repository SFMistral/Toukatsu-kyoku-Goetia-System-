# -*- coding: utf-8 -*-
"""
位置编码模块

提供各种位置编码方式：正弦、可学习、旋转等。
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn


class SinusoidalPositionEncoding(nn.Module):
    """
    正弦位置编码（固定）
    
    Args:
        num_feats: 特征维度（实际输出维度为2*num_feats）
        temperature: 温度参数
        normalize: 是否归一化坐标
        offset: 坐标偏移量
    """
    
    def __init__(
        self,
        num_feats: int = 128,
        temperature: int = 10000,
        normalize: bool = False,
        offset: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.offset = offset
        self.eps = eps
        
    def forward(
        self,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mask: (B, H, W) 有效区域mask
            
        Returns:
            pos: (B, C, H, W) 位置编码
        """
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * 2 * math.pi
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * 2 * math.pi
            
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4)
        
        pos_x = pos_x.flatten(3)
        pos_y = pos_y.flatten(3)
        
        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)
        
        return pos


class LearnedPositionEncoding(nn.Module):
    """
    可学习位置编码
    
    Args:
        num_feats: 特征维度
        num_positions: 位置数量（序列长度）
    """
    
    def __init__(
        self,
        num_feats: int = 256,
        num_positions: int = 50,
    ):
        super().__init__()
        self.num_feats = num_feats
        self.num_positions = num_positions
        
        self.row_embed = nn.Embedding(num_positions, num_feats)
        self.col_embed = nn.Embedding(num_positions, num_feats)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        
    def forward(
        self,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mask: (B, H, W) 有效区域mask
            
        Returns:
            pos: (B, C, H, W) 位置编码
        """
        H, W = mask.shape[-2:]
        
        i = torch.arange(W, device=mask.device)
        j = torch.arange(H, device=mask.device)
        
        x_embed = self.col_embed(i)  # (W, C/2)
        y_embed = self.row_embed(j)  # (H, C/2)
        
        pos = torch.cat([
            x_embed.unsqueeze(0).repeat(H, 1, 1),
            y_embed.unsqueeze(1).repeat(1, W, 1),
        ], dim=-1)  # (H, W, C)
        
        pos = pos.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        pos = pos.repeat(mask.shape[0], 1, 1, 1)  # (B, C, H, W)
        
        return pos


class RotaryPositionEncoding(nn.Module):
    """
    旋转位置编码（RoPE）
    
    Args:
        dim: 特征维度
        max_seq_len: 最大序列长度
        base: 基数
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算cos和sin
        self._set_cos_sin_cache(max_seq_len)
        
    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
        
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, heads, N, head_dim)
            seq_len: 序列长度
            
        Returns:
            (cos, sin) 用于旋转
        """
        if seq_len is None:
            seq_len = x.shape[2]
            
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
            
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )


class RelativePositionBias(nn.Module):
    """
    相对位置偏置（Swin Transformer）
    
    Args:
        window_size: 窗口大小
        num_heads: 注意力头数
    """
    
    def __init__(
        self,
        window_size: Tuple[int, int],
        num_heads: int,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        
        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # 计算相对位置索引
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def forward(self) -> torch.Tensor:
        """
        Returns:
            relative_position_bias: (num_heads, N, N)
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )
        return relative_position_bias.permute(2, 0, 1).contiguous()
