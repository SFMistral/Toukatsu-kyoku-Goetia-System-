# -*- coding: utf-8 -*-
"""
任务头模块

提供分类、检测、分割等任务头。
"""

from .cls_head import LinearClsHead, StackedLinearClsHead, VisionTransformerClsHead
from .det_head import AnchorHead, YOLOHead, FCOSHead
from .seg_head import FCNHead, ASPPHead, PSPHead, UPerHead, SegFormerHead
from .inst_seg_head import MaskRCNNHead

from registry.model_registry import HEADS

__all__ = [
    'LinearClsHead', 'StackedLinearClsHead', 'VisionTransformerClsHead',
    'AnchorHead', 'YOLOHead', 'FCOSHead',
    'FCNHead', 'ASPPHead', 'PSPHead', 'UPerHead', 'SegFormerHead',
    'MaskRCNNHead',
    'HEADS',
]
