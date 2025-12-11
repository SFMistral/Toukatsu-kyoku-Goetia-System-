# -*- coding: utf-8 -*-
"""
Models模块 - 深度学习模型组件库

提供模块化、可组合的深度学习模型构建能力。
采用Backbone-Neck-Head架构范式，支持分类、检测、分割三大任务。

主要组件：
- backbones: 骨干网络（ResNet, EfficientNet, ViT, Swin等）
- necks: 特征融合层（FPN, PAN, BiFPN等）
- heads: 任务头（分类头、检测头、分割头）
- detectors: 检测器（Faster R-CNN, YOLO, FCOS, DETR等）
- segmentors: 分割器（U-Net, DeepLabV3, SegFormer等）
- classifiers: 分类器（ImageClassifier）
- layers: 基础层组件
- utils: 模型工具

使用示例：
    from components.models import build_model, build_backbone
    
    # 构建完整模型
    model = build_model(config.model)
    
    # 构建骨干网络
    backbone = build_backbone(dict(type='ResNet', depth=50))
    
    # 模型前向
    losses = model(images, targets, mode='loss')
    predictions = model(images, mode='predict')
"""

from .builder import build_model, build_backbone, build_neck, build_head
from .base_model import BaseModel

# Backbones
from .backbones import (
    ResNet, ResNeXt, Res2Net,
    EfficientNet, EfficientNetV2,
    VisionTransformer,
    SwinTransformer,
    ConvNeXt,
    MobileNetV2, MobileNetV3,
    Darknet, CSPDarknet,
    BACKBONES,
)

# Necks
from .necks import FPN, PAN, BiFPN, YOLONeck, NECKS

# Heads
from .heads import (
    LinearClsHead, StackedLinearClsHead, VisionTransformerClsHead,
    AnchorHead, YOLOHead, FCOSHead,
    FCNHead, ASPPHead, PSPHead, UPerHead, SegFormerHead,
    MaskRCNNHead,
    HEADS,
)


# Classifiers
from .classifiers import BaseClassifier, ImageClassifier

# Detectors
from .detectors import (
    BaseDetector,
    FasterRCNN,
    YOLO,
    FCOS,
    DETR,
    RetinaNet,
)

# Segmentors
from .segmentors import (
    BaseSegmentor,
    UNet,
    DeepLabV3, DeepLabV3Plus,
    SegFormer,
    PSPNet,
    MaskRCNN,
)

# Layers
from .layers import (
    ConvModule, DepthwiseSeparableConv,
    build_norm_layer, build_activation,
    SELayer, CBAM, ECALayer, MultiHeadAttention,
    DropPath, DropBlock2d,
    BasicBlock, Bottleneck, InvertedResidual,
    TransformerEncoderLayer, TransformerDecoderLayer,
)

# Utils
from .utils import (
    init_weights, constant_init, xavier_init, kaiming_init,
    get_model_complexity, count_parameters,
    freeze_module, unfreeze_module, fuse_conv_bn,
    load_checkpoint, auto_convert,
)

# Registry
from registry.model_registry import MODELS

__all__ = [
    # Builder
    'build_model', 'build_backbone', 'build_neck', 'build_head',
    # Base
    'BaseModel',
    # Backbones
    'ResNet', 'ResNeXt', 'Res2Net',
    'EfficientNet', 'EfficientNetV2',
    'VisionTransformer', 'SwinTransformer', 'ConvNeXt',
    'MobileNetV2', 'MobileNetV3',
    'Darknet', 'CSPDarknet',
    'BACKBONES',
    # Necks
    'FPN', 'PAN', 'BiFPN', 'YOLONeck', 'NECKS',
    # Heads
    'LinearClsHead', 'StackedLinearClsHead', 'VisionTransformerClsHead',
    'AnchorHead', 'YOLOHead', 'FCOSHead',
    'FCNHead', 'ASPPHead', 'PSPHead', 'UPerHead', 'SegFormerHead',
    'MaskRCNNHead', 'HEADS',
    # Classifiers
    'BaseClassifier', 'ImageClassifier',
    # Detectors
    'BaseDetector', 'FasterRCNN', 'YOLO', 'FCOS', 'DETR', 'RetinaNet',
    # Segmentors
    'BaseSegmentor', 'UNet', 'DeepLabV3', 'DeepLabV3Plus',
    'SegFormer', 'PSPNet', 'MaskRCNN',
    # Layers
    'ConvModule', 'DepthwiseSeparableConv',
    'build_norm_layer', 'build_activation',
    'SELayer', 'CBAM', 'ECALayer', 'MultiHeadAttention',
    'DropPath', 'DropBlock2d',
    'BasicBlock', 'Bottleneck', 'InvertedResidual',
    'TransformerEncoderLayer', 'TransformerDecoderLayer',
    # Utils
    'init_weights', 'constant_init', 'xavier_init', 'kaiming_init',
    'get_model_complexity', 'count_parameters',
    'freeze_module', 'unfreeze_module', 'fuse_conv_bn',
    'load_checkpoint', 'auto_convert',
    # Registry
    'MODELS',
]
