# -*- coding: utf-8 -*-
"""
Vision Transformer骨干网络

支持ViT-Tiny/Small/Base/Large/Huge。
"""

from typing import Dict, Any, Optional, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.model_registry import BACKBONES
from ..layers import DropPath, TransformerEncoderLayer


VIT_CONFIGS = {
    'tiny': {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
    'small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
    'base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
    'large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
    'huge': {'embed_dim': 1280, 'depth': 32, 'num_heads': 16},
}


class PatchEmbed(nn.Module):
    """图像Patch嵌入"""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)  # (B, C, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


@BACKBONES.register(name='VisionTransformer')
class VisionTransformer(nn.Module):
    """
    Vision Transformer骨干网络
    
    Args:
        arch: 架构 tiny/small/base/large/huge
        img_size: 输入图像尺寸
        patch_size: Patch尺寸
        in_channels: 输入通道数
        out_indices: 输出层索引
        drop_rate: Dropout概率
        drop_path_rate: DropPath概率
        with_cls_token: 是否使用CLS token
        output_cls_token: 是否输出CLS token
    """

    def __init__(
        self,
        arch: str = 'base',
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        out_indices: Tuple[int, ...] = (-1,),
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        with_cls_token: bool = True,
        output_cls_token: bool = False,
        mlp_ratio: float = 4.0,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        if arch not in VIT_CONFIGS:
            raise ValueError(f"Invalid arch {arch}")
            
        config = VIT_CONFIGS[arch]
        embed_dim = config['embed_dim']
        depth = config['depth']
        num_heads = config['num_heads']
        
        self.embed_dim = embed_dim
        self.out_indices = out_indices
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + (1 if with_cls_token else 0), embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=drop_rate,
                drop_path=dpr[i],
            ) for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.out_channels = [embed_dim] * len(out_indices)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.with_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
        # Add position embedding
        x = x + self._resize_pos_embed(x.shape[1])
        x = self.pos_drop(x)
        
        # Transformer blocks
        outs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.out_indices or (i - len(self.blocks)) in self.out_indices:
                out = self.norm(x)
                if self.with_cls_token and not self.output_cls_token:
                    out = out[:, 1:]
                outs.append(out)
                
        return outs
        
    def _resize_pos_embed(self, num_tokens: int) -> torch.Tensor:
        """调整位置编码大小"""
        if num_tokens == self.pos_embed.shape[1]:
            return self.pos_embed
            
        # 需要插值
        if self.with_cls_token:
            cls_pos = self.pos_embed[:, :1]
            patch_pos = self.pos_embed[:, 1:]
        else:
            patch_pos = self.pos_embed
            
        # 简化处理：直接截断或填充
        target_len = num_tokens - (1 if self.with_cls_token else 0)
        if patch_pos.shape[1] >= target_len:
            patch_pos = patch_pos[:, :target_len]
        else:
            padding = torch.zeros(1, target_len - patch_pos.shape[1], self.embed_dim, device=patch_pos.device)
            patch_pos = torch.cat([patch_pos, padding], dim=1)
            
        if self.with_cls_token:
            return torch.cat([cls_pos, patch_pos], dim=1)
        return patch_pos
