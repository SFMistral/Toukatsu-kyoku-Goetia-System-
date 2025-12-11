# -*- coding: utf-8 -*-
"""
模型工具模块

提供权重初始化、模型工具函数、检查点转换等功能。
"""

from .weight_init import (
    constant_init,
    xavier_init,
    kaiming_init,
    normal_init,
    uniform_init,
    trunc_normal_init,
    bias_init_with_prob,
    init_weights,
)
from .model_utils import (
    get_model_complexity,
    count_parameters,
    freeze_module,
    unfreeze_module,
    freeze_bn,
    set_bn_eval,
    fuse_conv_bn,
    replace_module,
    get_module_by_name,
    make_divisible,
    auto_pad,
)
from .ckpt_convert import (
    convert_checkpoint,
    convert_from_torchvision,
    convert_from_timm,
    auto_convert,
    remap_keys,
    filter_by_prefix,
    strip_prefix,
    add_prefix,
    load_checkpoint,
)

__all__ = [
    # weight_init
    'constant_init', 'xavier_init', 'kaiming_init', 'normal_init',
    'uniform_init', 'trunc_normal_init', 'bias_init_with_prob', 'init_weights',
    # model_utils
    'get_model_complexity', 'count_parameters', 'freeze_module', 'unfreeze_module',
    'freeze_bn', 'set_bn_eval', 'fuse_conv_bn', 'replace_module',
    'get_module_by_name', 'make_divisible', 'auto_pad',
    # ckpt_convert
    'convert_checkpoint', 'convert_from_torchvision', 'convert_from_timm',
    'auto_convert', 'remap_keys', 'filter_by_prefix', 'strip_prefix',
    'add_prefix', 'load_checkpoint',
]
