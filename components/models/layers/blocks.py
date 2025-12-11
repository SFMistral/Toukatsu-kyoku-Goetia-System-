# -*- coding: utf-8 -*-
"""
复合模块块

提供各种复合模块：ResNet块、MobileNet块、Transformer块等。
"""

from typing import Dict, Any, Optional, List, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_module import ConvModule, auto_pad
from .norm_layers import build_norm_layer
from .activation import build_activation
from .drop import DropPath


class BasicBlock(nn.Module):
    """
    ResNet基础块（2层卷积）
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 步长
        dilation: 空洞率
        downsample: 下采样层
        norm_cfg: 归一化配置
        act_cfg: 激活配置
        drop_path_rate: DropPath概率
    """
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='ReLU', inplace=True),
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        
        self.conv1 = ConvModule(
            in_channels, out_channels, 3,
            stride=stride, padding=dilation, dilation=dilation,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            out_channels, out_channels, 3,
            padding=1,
            norm_cfg=norm_cfg, act_cfg=None,
        )
        
        self.downsample = downsample
        self.act = build_activation(act_cfg)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.drop_path(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.act(out)
        
        return out


class Bottleneck(nn.Module):
    """
    ResNet瓶颈块（3层卷积）
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 步长
        dilation: 空洞率
        downsample: 下采样层
        groups: 分组数（ResNeXt）
        base_width: 基础宽度
        norm_cfg: 归一化配置
        act_cfg: 激活配置
        drop_path_rate: DropPath概率
    """
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='ReLU', inplace=True),
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        
        width = int(out_channels * (base_width / 64.0)) * groups
        
        self.conv1 = ConvModule(
            in_channels, width, 1,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            width, width, 3,
            stride=stride, padding=dilation, dilation=dilation, groups=groups,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.conv3 = ConvModule(
            width, out_channels * self.expansion, 1,
            norm_cfg=norm_cfg, act_cfg=None,
        )
        
        self.downsample = downsample
        self.act = build_activation(act_cfg)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.drop_path(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.act(out)
        
        return out


class InvertedResidual(nn.Module):
    """
    MobileNet倒残差块
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 步长
        expand_ratio: 扩展比例
        norm_cfg: 归一化配置
        act_cfg: 激活配置
        use_se: 是否使用SE注意力
        se_ratio: SE压缩比例
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 6.0,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='ReLU6', inplace=True),
        use_se: bool = False,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        
        self.stride = stride
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        hidden_dim = int(round(in_channels * expand_ratio))
        
        layers = []
        
        # 扩展层（如果expand_ratio != 1）
        if expand_ratio != 1:
            layers.append(ConvModule(
                in_channels, hidden_dim, 1,
                norm_cfg=norm_cfg, act_cfg=act_cfg,
            ))
            
        # 深度卷积
        layers.append(ConvModule(
            hidden_dim, hidden_dim, 3,
            stride=stride, padding=1, groups=hidden_dim,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        ))
        
        # SE注意力
        if use_se:
            from .attention import SELayer
            layers.append(SELayer(hidden_dim, reduction=int(1 / se_ratio)))
            
        # 投影层
        layers.append(ConvModule(
            hidden_dim, out_channels, 1,
            norm_cfg=norm_cfg, act_cfg=None,
        ))
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class CSPLayer(nn.Module):
    """
    CSP层（YOLO系列）
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        num_blocks: 块数量
        expansion: 扩展比例
        norm_cfg: 归一化配置
        act_cfg: 激活配置
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        expansion: float = 0.5,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='SiLU', inplace=True),
    ):
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = ConvModule(
            in_channels, hidden_channels, 1,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            in_channels, hidden_channels, 1,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.conv3 = ConvModule(
            hidden_channels * 2, out_channels, 1,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        
        self.blocks = nn.Sequential(*[
            Bottleneck(
                hidden_channels, hidden_channels // 4,
                norm_cfg=norm_cfg, act_cfg=act_cfg,
            ) for _ in range(num_blocks)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = self.blocks(x1)
        return self.conv3(torch.cat([x1, x2], dim=1))


class C2f(nn.Module):
    """
    C2f模块（YOLOv8）
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        num_blocks: 块数量
        shortcut: 是否使用shortcut
        expansion: 扩展比例
        norm_cfg: 归一化配置
        act_cfg: 激活配置
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='SiLU', inplace=True),
    ):
        super().__init__()
        
        self.hidden_channels = int(out_channels * expansion)
        
        self.conv1 = ConvModule(
            in_channels, 2 * self.hidden_channels, 1,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            (2 + num_blocks) * self.hidden_channels, out_channels, 1,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        
        self.blocks = nn.ModuleList([
            _C2fBottleneck(
                self.hidden_channels, self.hidden_channels,
                shortcut=shortcut,
                norm_cfg=norm_cfg, act_cfg=act_cfg,
            ) for _ in range(num_blocks)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = list(x.chunk(2, dim=1))
        
        for block in self.blocks:
            x.append(block(x[-1]))
            
        return self.conv2(torch.cat(x, dim=1))


class _C2fBottleneck(nn.Module):
    """C2f内部瓶颈块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='SiLU', inplace=True),
    ):
        super().__init__()
        
        self.shortcut = shortcut and in_channels == out_channels
        
        self.conv1 = ConvModule(
            in_channels, out_channels, 3, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            out_channels, out_channels, 3, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return x + out if self.shortcut else out


class FFN(nn.Module):
    """
    前馈网络（Transformer）
    
    Args:
        embed_dim: 嵌入维度
        hidden_dim: 隐藏层维度
        dropout: Dropout概率
        act_cfg: 激活配置
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        act_cfg: Dict[str, Any] = dict(type='GELU'),
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or embed_dim * 4
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = build_activation(act_cfg)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """
    多层感知机
    
    Args:
        in_features: 输入特征数
        hidden_features: 隐藏层特征数
        out_features: 输出特征数
        dropout: Dropout概率
        act_cfg: 激活配置
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.0,
        act_cfg: Dict[str, Any] = dict(type='GELU'),
    ):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation(act_cfg)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码层
    
    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        mlp_ratio: MLP扩展比例
        dropout: Dropout概率
        attn_drop: 注意力Dropout
        drop_path: DropPath概率
        act_cfg: 激活配置
        norm_cfg: 归一化配置
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_cfg: Dict[str, Any] = dict(type='GELU'),
        norm_cfg: Dict[str, Any] = dict(type='LN'),
    ):
        super().__init__()
        
        from .attention import MultiHeadAttention
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim, num_heads,
            attn_drop=attn_drop, proj_drop=dropout,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            embed_dim, int(embed_dim * mlp_ratio),
            dropout=dropout, act_cfg=act_cfg,
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码层
    
    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        mlp_ratio: MLP扩展比例
        dropout: Dropout概率
        attn_drop: 注意力Dropout
        drop_path: DropPath概率
        act_cfg: 激活配置
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_cfg: Dict[str, Any] = dict(type='GELU'),
    ):
        super().__init__()
        
        from .attention import MultiHeadAttention, CrossAttention
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(
            embed_dim, num_heads,
            attn_drop=attn_drop, proj_drop=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(
            embed_dim, num_heads,
            attn_drop=attn_drop, proj_drop=dropout,
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            embed_dim, int(embed_dim * mlp_ratio),
            dropout=dropout, act_cfg=act_cfg,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.self_attn(self.norm1(x), self_attn_mask))
        x = x + self.drop_path(self.cross_attn(self.norm2(x), memory, cross_attn_mask))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class SPPF(nn.Module):
    """
    空间金字塔池化快速版（YOLOv5）
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 池化核大小
        norm_cfg: 归一化配置
        act_cfg: 激活配置
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='SiLU', inplace=True),
    ):
        super().__init__()
        
        hidden_channels = in_channels // 2
        
        self.conv1 = ConvModule(
            in_channels, hidden_channels, 1,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            hidden_channels * 4, out_channels, 1,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))
