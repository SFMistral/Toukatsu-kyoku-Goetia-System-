# -*- coding: utf-8 -*-
"""
分类器模块

提供图像分类器。
"""

from .base_classifier import BaseClassifier
from .image_classifier import ImageClassifier

from registry.model_registry import MODELS

__all__ = ['BaseClassifier', 'ImageClassifier', 'MODELS']
