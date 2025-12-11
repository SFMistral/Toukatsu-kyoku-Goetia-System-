# -*- coding: utf-8 -*-
"""
分割器模块

提供语义分割和实例分割器。
"""

from .base_segmentor import BaseSegmentor
from .unet import UNet
from .deeplabv3 import DeepLabV3, DeepLabV3Plus
from .segformer import SegFormer
from .pspnet import PSPNet
from .mask_rcnn import MaskRCNN

from registry.model_registry import MODELS

__all__ = [
    'BaseSegmentor', 'UNet', 'DeepLabV3', 'DeepLabV3Plus',
    'SegFormer', 'PSPNet', 'MaskRCNN', 'MODELS',
]
