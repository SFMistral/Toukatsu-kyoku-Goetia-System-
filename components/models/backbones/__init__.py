# -*- coding: utf-8 -*-
"""
骨干网络模块

提供各种骨干网络：ResNet、EfficientNet、ViT、Swin等。
"""

from .resnet import ResNet, ResNeXt, Res2Net
from .efficientnet import EfficientNet, EfficientNetV2
from .vit import VisionTransformer
from .swin_transformer import SwinTransformer
from .convnext import ConvNeXt
from .mobilenet import MobileNetV2, MobileNetV3
from .darknet import Darknet, CSPDarknet

from registry.model_registry import BACKBONES

__all__ = [
    'ResNet', 'ResNeXt', 'Res2Net',
    'EfficientNet', 'EfficientNetV2',
    'VisionTransformer',
    'SwinTransformer',
    'ConvNeXt',
    'MobileNetV2', 'MobileNetV3',
    'Darknet', 'CSPDarknet',
    'BACKBONES',
]
