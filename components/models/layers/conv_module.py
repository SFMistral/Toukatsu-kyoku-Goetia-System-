# -*- coding: utf-8 -*-
"""
卷积模块封装

提供Conv + Norm + Activation组合模块及各种卷积变体。
"""

from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm_layers import build_norm_layer
from .activation import build_activation


class ConvModule(nn.Module):
    """
    卷积模块：Conv + Norm + Activation 组合
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充，'auto'时自动计算same padding
        dilation: 空洞率
        groups: 分组数
        bias: 是否使用偏置，'auto'时根据norm自动决定
        norm_cfg: 归一化层配置
        act_cfg: 激活函数配置
        order: 组件顺序，默认('conv', 'norm', 'act')
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: Union[bool, str] = 'auto',
        norm_cfg: Optional[Dict[str, Any]] = None,
        act_cfg: Optional[Dict[str, Any]] = dict(type='ReLU', inplace=True),
        order: Tuple[str, ...] = ('conv', 'norm', 'act'),
    ):
        super().__init__()
        
        self.order = order
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        
        # 自动计算padding
        if padding == 'auto':
            padding = auto_pad(kernel_size, dilation)
            
        # 自动决定bias
        if bias == 'auto':
            bias = not self.with_norm
            
        # 构建卷积层
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        
        # 构建归一化层
        if self.with_norm:
            norm_name, self.norm = build_norm_layer(norm_cfg, out_channels)
            self.norm_name = norm_name
        else:
            self.norm = None
            
        # 构建激活函数
        if self.with_activation:
            self.activate = build_activation(act_cfg)
        else:
            self.activate = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_name in self.order:
            if layer_name == 'conv':
                x = self.conv(x)
            elif layer_name == 'norm' and self.norm is not None:
                x = self.norm(x)
            elif layer_name == 'act' and self.activate is not None:
                x = self.activate(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积：Depthwise Conv + Pointwise Conv
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 深度卷积核大小
        stride: 步长
        padding: 填充
        dilation: 空洞率
        norm_cfg: 归一化层配置
        act_cfg: 激活函数配置
        dw_norm_cfg: 深度卷积归一化配置，None时使用norm_cfg
        dw_act_cfg: 深度卷积激活配置，None时使用act_cfg
        pw_norm_cfg: 点卷积归一化配置，None时使用norm_cfg
        pw_act_cfg: 点卷积激活配置，None时使用act_cfg
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 'auto',
        dilation: Union[int, Tuple[int, int]] = 1,
        norm_cfg: Optional[Dict[str, Any]] = dict(type='BN'),
        act_cfg: Optional[Dict[str, Any]] = dict(type='ReLU', inplace=True),
        dw_norm_cfg: Optional[Dict[str, Any]] = None,
        dw_act_cfg: Optional[Dict[str, Any]] = None,
        pw_norm_cfg: Optional[Dict[str, Any]] = None,
        pw_act_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        # 深度卷积
        self.depthwise = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg or norm_cfg,
            act_cfg=dw_act_cfg or act_cfg,
        )
        
        # 点卷积
        self.pointwise = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg or norm_cfg,
            act_cfg=pw_act_cfg or act_cfg,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DeformConv2d(nn.Module):
    """
    可变形卷积 v1
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        dilation: 空洞率
        groups: 分组数
        deform_groups: 可变形分组数
        bias: 是否使用偏置
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        
        # offset卷积：预测每个采样点的偏移
        self.offset_conv = nn.Conv2d(
            in_channels,
            deform_groups * 2 * kernel_size * kernel_size,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        
        # 主卷积权重
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self._init_weights()
        
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        # 使用torchvision的deform_conv2d
        try:
            from torchvision.ops import deform_conv2d
            return deform_conv2d(
                x, offset, self.weight, self.bias,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation
            )
        except ImportError:
            # 回退到普通卷积
            return F.conv2d(
                x, self.weight, self.bias,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
            )


class ModulatedDeformConv2d(nn.Module):
    """
    调制可变形卷积 v2
    
    相比v1增加了调制因子，控制每个采样点的权重。
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        
        # offset + mask卷积
        self.offset_conv = nn.Conv2d(
            in_channels,
            deform_groups * 3 * kernel_size * kernel_size,  # 2 for offset, 1 for mask
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self._init_weights()
        
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.offset_conv(x)
        k2 = self.kernel_size * self.kernel_size
        offset = out[:, :self.deform_groups * 2 * k2]
        mask = out[:, self.deform_groups * 2 * k2:].sigmoid()
        
        try:
            from torchvision.ops import deform_conv2d
            return deform_conv2d(
                x, offset, self.weight, self.bias,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, mask=mask
            )
        except ImportError:
            return F.conv2d(
                x, self.weight, self.bias,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
            )


def auto_pad(
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]] = 1
) -> Union[int, Tuple[int, int]]:
    """自动计算same padding"""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
        
    pad_h = (kernel_size[0] - 1) * dilation[0] // 2
    pad_w = (kernel_size[1] - 1) * dilation[1] // 2
    
    if pad_h == pad_w:
        return pad_h
    return (pad_h, pad_w)
