# -*- coding: utf-8 -*-
"""
基础层组件模块

提供卷积模块、归一化层、激活函数、注意力机制等基础组件。
"""

from .conv_module import (
    ConvModule,
    DepthwiseSeparableConv,
    DeformConv2d,
    ModulatedDeformConv2d,
)
from .norm_layers import (
    build_norm_layer,
    NORM_LAYERS,
    LayerNorm2d,
)
from .activation import (
    build_activation,
    ACTIVATIONS,
    Swish,
    Mish,
    HardSwish,
    HardSigmoid,
)
from .attention import (
    SELayer,
    CBAM,
    ECALayer,
    MultiHeadAttention,
    WindowAttention,
    CrossAttention,
    SpatialAttention,
    ChannelAttention,
)
from .drop import (
    Dropout,
    DropPath,
    DropBlock2d,
)
from .position_encoding import (
    SinusoidalPositionEncoding,
    LearnedPositionEncoding,
    RotaryPositionEncoding,
    RelativePositionBias,
)
from .blocks import (
    BasicBlock,
    Bottleneck,
    InvertedResidual,
    CSPLayer,
    C2f,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    FFN,
    MLP,
    SPPF,
)

__all__ = [
    # conv_module
    'ConvModule', 'DepthwiseSeparableConv', 'DeformConv2d', 'ModulatedDeformConv2d',
    # norm_layers
    'build_norm_layer', 'NORM_LAYERS', 'LayerNorm2d',
    # activation
    'build_activation', 'ACTIVATIONS', 'Swish', 'Mish', 'HardSwish', 'HardSigmoid',
    # attention
    'SELayer', 'CBAM', 'ECALayer', 'MultiHeadAttention', 'WindowAttention',
    'CrossAttention', 'SpatialAttention', 'ChannelAttention',
    # drop
    'Dropout', 'DropPath', 'DropBlock2d',
    # position_encoding
    'SinusoidalPositionEncoding', 'LearnedPositionEncoding',
    'RotaryPositionEncoding', 'RelativePositionBias',
    # blocks
    'BasicBlock', 'Bottleneck', 'InvertedResidual', 'CSPLayer', 'C2f',
    'TransformerEncoderLayer', 'TransformerDecoderLayer', 'FFN', 'MLP', 'SPPF',
]
