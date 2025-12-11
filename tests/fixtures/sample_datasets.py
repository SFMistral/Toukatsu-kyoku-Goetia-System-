"""
示例数据集夹具
"""
import pytest
import numpy as np
from typing import Dict, List, Any


@pytest.fixture
def sample_image_data():
    """示例图像数据"""
    return {
        'images': np.random.rand(10, 3, 224, 224).astype(np.float32),
        'labels': np.random.randint(0, 10, size=10),
        'num_classes': 10
    }


@pytest.fixture
def sample_classification_dataset():
    """示例分类数据集配置"""
    return {
        'name': 'test_classification',
        'type': 'classification',
        'train_size': 1000,
        'val_size': 200,
        'test_size': 200,
        'num_classes': 10,
        'image_size': (224, 224),
        'channels': 3
    }


@pytest.fixture
def sample_detection_dataset():
    """示例检测数据集配置"""
    return {
        'name': 'test_detection',
        'type': 'detection',
        'train_size': 500,
        'val_size': 100,
        'num_classes': 20,
        'image_size': (640, 640),
        'annotation_format': 'coco'
    }


@pytest.fixture
def sample_segmentation_dataset():
    """示例分割数据集配置"""
    return {
        'name': 'test_segmentation',
        'type': 'segmentation',
        'train_size': 300,
        'val_size': 50,
        'num_classes': 21,
        'image_size': (512, 512),
        'mask_format': 'png'
    }


@pytest.fixture
def sample_batch_data():
    """示例批次数据"""
    batch_size = 4
    return {
        'images': np.random.rand(batch_size, 3, 224, 224).astype(np.float32),
        'labels': np.random.randint(0, 10, size=batch_size),
        'batch_size': batch_size
    }