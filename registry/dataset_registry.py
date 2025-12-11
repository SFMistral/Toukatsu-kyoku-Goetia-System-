# -*- coding: utf-8 -*-
"""
数据集注册器模块

管理数据集相关组件：数据集类、采样器、批处理函数等。
"""

from typing import Dict, Any, Optional, List, Type
from .registry import Registry, ComponentSource


class DatasetRegistry(Registry):
    """数据集注册器，支持元信息查询和兼容性检查"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._dataset_meta: Dict[str, Dict[str, Any]] = {}
        
    def register(
        self,
        name: Optional[str] = None,
        cls: Optional[Type] = None,
        num_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        task_type: Optional[str] = None,
        **kwargs
    ):
        """
        注册数据集组件
        
        Args:
            name: 组件名称
            cls: 组件类
            num_classes: 类别数量
            class_names: 类别名称列表
            task_type: 任务类型 (classification, detection, segmentation等)
            **kwargs: 其他注册参数
        """
        result = super().register(name=name, cls=cls, **kwargs)
        
        # 记录数据集元信息
        actual_name = name or (cls.__name__ if cls else None)
        if actual_name:
            self._dataset_meta[actual_name] = {
                'num_classes': num_classes,
                'class_names': class_names,
                'task_type': task_type
            }
            
        return result
        
    def get_meta_info(self, name: str) -> Dict[str, Any]:
        """获取数据集元信息"""
        if name in self._dataset_meta:
            return self._dataset_meta[name]
        raise KeyError(f"Dataset '{name}' meta info not found")
        
    def get_num_classes(self, name: str) -> Optional[int]:
        """获取数据集类别数"""
        return self._dataset_meta.get(name, {}).get('num_classes')
        
    def get_class_names(self, name: str) -> Optional[List[str]]:
        """获取数据集类别名称"""
        return self._dataset_meta.get(name, {}).get('class_names')
        
    def get_task_type(self, name: str) -> Optional[str]:
        """获取数据集任务类型"""
        return self._dataset_meta.get(name, {}).get('task_type')
        
    def check_compatibility(self, dataset_name: str, model_task_type: str) -> bool:
        """
        检查数据集与模型任务类型的兼容性
        
        Args:
            dataset_name: 数据集名称
            model_task_type: 模型任务类型
            
        Returns:
            是否兼容
        """
        dataset_task = self.get_task_type(dataset_name)
        if dataset_task is None:
            return True  # 未指定任务类型时默认兼容
        return dataset_task == model_task_type
        
    def list_by_task(self, task_type: str) -> List[str]:
        """按任务类型筛选数据集"""
        return [
            name for name, meta in self._dataset_meta.items()
            if meta.get('task_type') == task_type
        ]


# 创建数据集相关注册器单例
DATASETS = DatasetRegistry('datasets', base_class=None)
SAMPLERS = Registry('samplers', base_class=None)
COLLATE_FNS = Registry('collate_fns', base_class=None)


def _register_builtin_datasets():
    """注册内置数据集"""
    try:
        from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100, MNIST
        
        DATASETS.register(
            name='ImageFolder',
            cls=ImageFolder,
            task_type='classification',
            source=ComponentSource.THIRD_PARTY
        )
        
        DATASETS.register(
            name='CIFAR10',
            cls=CIFAR10,
            num_classes=10,
            class_names=['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck'],
            task_type='classification',
            source=ComponentSource.THIRD_PARTY
        )
        
        DATASETS.register(
            name='CIFAR100',
            cls=CIFAR100,
            num_classes=100,
            task_type='classification',
            source=ComponentSource.THIRD_PARTY
        )
        
        DATASETS.register(
            name='MNIST',
            cls=MNIST,
            num_classes=10,
            class_names=[str(i) for i in range(10)],
            task_type='classification',
            source=ComponentSource.THIRD_PARTY
        )
    except ImportError:
        pass


def _register_builtin_samplers():
    """注册内置采样器"""
    try:
        from torch.utils.data import (
            RandomSampler, SequentialSampler, 
            WeightedRandomSampler, SubsetRandomSampler
        )
        from torch.utils.data.distributed import DistributedSampler
        
        SAMPLERS.register('RandomSampler', RandomSampler, source=ComponentSource.THIRD_PARTY)
        SAMPLERS.register('SequentialSampler', SequentialSampler, source=ComponentSource.THIRD_PARTY)
        SAMPLERS.register('WeightedRandomSampler', WeightedRandomSampler, source=ComponentSource.THIRD_PARTY)
        SAMPLERS.register('SubsetRandomSampler', SubsetRandomSampler, source=ComponentSource.THIRD_PARTY)
        SAMPLERS.register('DistributedSampler', DistributedSampler, source=ComponentSource.THIRD_PARTY)
    except ImportError:
        pass
