# -*- coding: utf-8 -*-
"""
检测器模块

提供各种目标检测器。
"""

from .base_detector import BaseDetector
from .faster_rcnn import FasterRCNN
from .yolo import YOLO
from .fcos import FCOS
from .detr import DETR
from .retinanet import RetinaNet

from registry.model_registry import MODELS

__all__ = [
    'BaseDetector', 'FasterRCNN', 'YOLO', 'FCOS', 'DETR', 'RetinaNet',
    'MODELS',
]
