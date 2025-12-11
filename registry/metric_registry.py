# -*- coding: utf-8 -*-
"""
评估指标注册器模块

管理各类评估指标：分类指标、检测指标、分割指标等。
支持指标累积计算和分布式同步。
"""

from typing import Dict, Any, Optional, List, Type, Union
from abc import ABC, abstractmethod
from .registry import Registry, ComponentSource


class MetricRegistry(Registry):
    """评估指标注册器"""
    
    def build_group(self, configs: List[Dict[str, Any]]) -> 'MetricGroup':
        """
        构建指标组
        
        Args:
            configs: 指标配置列表
            
        Returns:
            指标组实例
        """
        metrics = {}
        for cfg in configs:
            cfg = cfg.copy()
            name = cfg.pop('name', cfg['type'])
            metrics[name] = self.build(cfg)
        return MetricGroup(metrics)


class MetricGroup:
    """指标组，管理多个指标的统一计算"""
    
    def __init__(self, metrics: Dict[str, 'BaseMetric']):
        self.metrics = metrics
        
    def update(self, pred, target, **kwargs):
        """更新所有指标"""
        for metric in self.metrics.values():
            metric.update(pred, target, **kwargs)
            
    def compute(self) -> Dict[str, float]:
        """计算所有指标"""
        return {name: metric.compute() for name, metric in self.metrics.items()}
        
    def reset(self):
        """重置所有指标"""
        for metric in self.metrics.values():
            metric.reset()
            
    def sync(self):
        """分布式同步所有指标"""
        for metric in self.metrics.values():
            if hasattr(metric, 'sync'):
                metric.sync()


# 创建评估指标注册器单例
METRICS = MetricRegistry('metrics', base_class=None)


class BaseMetric(ABC):
    """评估指标基类"""
    
    def __init__(self):
        self.reset()
        
    @abstractmethod
    def update(self, pred, target, **kwargs):
        """更新指标状态"""
        pass
        
    @abstractmethod
    def compute(self) -> float:
        """计算指标值"""
        pass
        
    @abstractmethod
    def reset(self):
        """重置指标状态"""
        pass
        
    def sync(self):
        """分布式同步（可选实现）"""
        pass


class Accuracy(BaseMetric):
    """准确率指标"""
    
    def __init__(self, topk: int = 1):
        self.topk = topk
        super().__init__()
        
    def reset(self):
        self.correct = 0
        self.total = 0
        
    def update(self, pred, target, **kwargs):
        import torch
        
        if self.topk == 1:
            pred_labels = pred.argmax(dim=-1)
            self.correct += (pred_labels == target).sum().item()
        else:
            _, pred_topk = pred.topk(self.topk, dim=-1)
            target_expand = target.unsqueeze(-1).expand_as(pred_topk)
            self.correct += (pred_topk == target_expand).any(dim=-1).sum().item()
            
        self.total += target.size(0)
        
    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class Precision(BaseMetric):
    """精确率指标"""
    
    def __init__(self, num_classes: int, average: str = 'macro'):
        self.num_classes = num_classes
        self.average = average
        super().__init__()
        
    def reset(self):
        self.tp = [0] * self.num_classes
        self.fp = [0] * self.num_classes
        
    def update(self, pred, target, **kwargs):
        pred_labels = pred.argmax(dim=-1)
        
        for c in range(self.num_classes):
            pred_c = (pred_labels == c)
            target_c = (target == c)
            self.tp[c] += (pred_c & target_c).sum().item()
            self.fp[c] += (pred_c & ~target_c).sum().item()
            
    def compute(self) -> float:
        precisions = []
        for c in range(self.num_classes):
            if self.tp[c] + self.fp[c] > 0:
                precisions.append(self.tp[c] / (self.tp[c] + self.fp[c]))
            else:
                precisions.append(0.0)
                
        if self.average == 'macro':
            return sum(precisions) / len(precisions)
        elif self.average == 'micro':
            total_tp = sum(self.tp)
            total_fp = sum(self.fp)
            return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        return precisions


class Recall(BaseMetric):
    """召回率指标"""
    
    def __init__(self, num_classes: int, average: str = 'macro'):
        self.num_classes = num_classes
        self.average = average
        super().__init__()
        
    def reset(self):
        self.tp = [0] * self.num_classes
        self.fn = [0] * self.num_classes
        
    def update(self, pred, target, **kwargs):
        pred_labels = pred.argmax(dim=-1)
        
        for c in range(self.num_classes):
            pred_c = (pred_labels == c)
            target_c = (target == c)
            self.tp[c] += (pred_c & target_c).sum().item()
            self.fn[c] += (~pred_c & target_c).sum().item()
            
    def compute(self) -> float:
        recalls = []
        for c in range(self.num_classes):
            if self.tp[c] + self.fn[c] > 0:
                recalls.append(self.tp[c] / (self.tp[c] + self.fn[c]))
            else:
                recalls.append(0.0)
                
        if self.average == 'macro':
            return sum(recalls) / len(recalls)
        elif self.average == 'micro':
            total_tp = sum(self.tp)
            total_fn = sum(self.fn)
            return total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        return recalls


class F1Score(BaseMetric):
    """F1分数指标"""
    
    def __init__(self, num_classes: int, average: str = 'macro'):
        self.num_classes = num_classes
        self.average = average
        self.precision = Precision(num_classes, average)
        self.recall = Recall(num_classes, average)
        super().__init__()
        
    def reset(self):
        self.precision.reset()
        self.recall.reset()
        
    def update(self, pred, target, **kwargs):
        self.precision.update(pred, target, **kwargs)
        self.recall.update(pred, target, **kwargs)
        
    def compute(self) -> float:
        p = self.precision.compute()
        r = self.recall.compute()
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


class MeanIoU(BaseMetric):
    """平均交并比指标（分割任务）"""
    
    def __init__(self, num_classes: int, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        super().__init__()
        
    def reset(self):
        self.intersection = [0] * self.num_classes
        self.union = [0] * self.num_classes
        
    def update(self, pred, target, **kwargs):
        pred_labels = pred.argmax(dim=1)
        
        mask = target != self.ignore_index
        pred_labels = pred_labels[mask]
        target = target[mask]
        
        for c in range(self.num_classes):
            pred_c = (pred_labels == c)
            target_c = (target == c)
            self.intersection[c] += (pred_c & target_c).sum().item()
            self.union[c] += (pred_c | target_c).sum().item()
            
    def compute(self) -> float:
        ious = []
        for c in range(self.num_classes):
            if self.union[c] > 0:
                ious.append(self.intersection[c] / self.union[c])
        return sum(ious) / len(ious) if ious else 0.0


class ConfusionMatrix(BaseMetric):
    """混淆矩阵"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        super().__init__()
        
    def reset(self):
        self.matrix = [[0] * self.num_classes for _ in range(self.num_classes)]
        
    def update(self, pred, target, **kwargs):
        pred_labels = pred.argmax(dim=-1)
        
        for p, t in zip(pred_labels.view(-1).tolist(), target.view(-1).tolist()):
            if 0 <= p < self.num_classes and 0 <= t < self.num_classes:
                self.matrix[t][p] += 1
                
    def compute(self):
        return self.matrix


# 注册内置指标
METRICS.register('Accuracy', Accuracy, category='classification', source=ComponentSource.BUILTIN)
METRICS.register('Precision', Precision, category='classification', source=ComponentSource.BUILTIN)
METRICS.register('Recall', Recall, category='classification', source=ComponentSource.BUILTIN)
METRICS.register('F1Score', F1Score, category='classification', source=ComponentSource.BUILTIN)
METRICS.register('MeanIoU', MeanIoU, category='segmentation', source=ComponentSource.BUILTIN)
METRICS.register('ConfusionMatrix', ConfusionMatrix, category='classification', source=ComponentSource.BUILTIN)
