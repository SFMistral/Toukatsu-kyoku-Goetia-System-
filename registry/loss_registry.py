# -*- coding: utf-8 -*-
"""
损失函数注册器模块

管理各类损失函数：分类损失、检测损失、分割损失等。
支持损失函数组合和动态权重调整。
"""

from typing import Dict, Any, Optional, List, Type, Union
from .registry import Registry, ComponentSource


class LossRegistry(Registry):
    """损失函数注册器，支持损失组合和动态权重"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        
    def build_combined(
        self,
        configs: List[Dict[str, Any]],
        weights: Optional[List[float]] = None
    ) -> 'CombinedLoss':
        """
        构建组合损失函数
        
        Args:
            configs: 损失函数配置列表
            weights: 各损失函数权重
            
        Returns:
            组合损失函数实例
        """
        losses = [self.build(cfg) for cfg in configs]
        
        if weights is None:
            weights = [1.0] * len(losses)
            
        return CombinedLoss(losses, weights)


class CombinedLoss:
    """组合损失函数"""
    
    def __init__(self, losses: List, weights: List[float]):
        self.losses = losses
        self.weights = weights
        self._dynamic_weights = list(weights)
        
    def __call__(self, *args, **kwargs):
        """计算加权损失和"""
        total_loss = 0
        loss_dict = {}
        
        for i, (loss_fn, weight) in enumerate(zip(self.losses, self._dynamic_weights)):
            loss = loss_fn(*args, **kwargs)
            loss_name = getattr(loss_fn, '__class__', type(loss_fn)).__name__
            loss_dict[f'loss_{i}_{loss_name}'] = loss.item() if hasattr(loss, 'item') else loss
            total_loss = total_loss + weight * loss
            
        loss_dict['total_loss'] = total_loss.item() if hasattr(total_loss, 'item') else total_loss
        return total_loss, loss_dict
        
    def update_weights(self, weights: List[float]):
        """动态更新权重"""
        if len(weights) != len(self.losses):
            raise ValueError(f"Expected {len(self.losses)} weights, got {len(weights)}")
        self._dynamic_weights = list(weights)
        
    def reset_weights(self):
        """重置为初始权重"""
        self._dynamic_weights = list(self.weights)


# 创建损失函数注册器单例
LOSSES = LossRegistry('losses', base_class=None)


def _register_builtin_losses():
    """注册内置损失函数"""
    try:
        import torch.nn as nn
        
        # 分类损失
        LOSSES.register('CrossEntropyLoss', nn.CrossEntropyLoss, 
                       category='classification', source=ComponentSource.THIRD_PARTY)
        LOSSES.register('BCELoss', nn.BCELoss,
                       category='classification', source=ComponentSource.THIRD_PARTY)
        LOSSES.register('BCEWithLogitsLoss', nn.BCEWithLogitsLoss,
                       category='classification', source=ComponentSource.THIRD_PARTY)
        LOSSES.register('NLLLoss', nn.NLLLoss,
                       category='classification', source=ComponentSource.THIRD_PARTY)
        
        # 回归损失
        LOSSES.register('L1Loss', nn.L1Loss,
                       category='regression', source=ComponentSource.THIRD_PARTY)
        LOSSES.register('MSELoss', nn.MSELoss,
                       category='regression', source=ComponentSource.THIRD_PARTY)
        LOSSES.register('SmoothL1Loss', nn.SmoothL1Loss,
                       category='regression', source=ComponentSource.THIRD_PARTY)
        LOSSES.register('HuberLoss', nn.HuberLoss,
                       category='regression', source=ComponentSource.THIRD_PARTY)
        
        # 其他损失
        LOSSES.register('KLDivLoss', nn.KLDivLoss,
                       category='other', source=ComponentSource.THIRD_PARTY)
        LOSSES.register('CosineEmbeddingLoss', nn.CosineEmbeddingLoss,
                       category='other', source=ComponentSource.THIRD_PARTY)
        LOSSES.register('TripletMarginLoss', nn.TripletMarginLoss,
                       category='other', source=ComponentSource.THIRD_PARTY)
    except ImportError:
        pass


# 自定义损失函数基类
class BaseLoss:
    """损失函数基类"""
    
    def __init__(self, reduction: str = 'mean'):
        self.reduction = reduction
        
    def __call__(self, pred, target):
        raise NotImplementedError
        
    def _reduce(self, loss):
        """应用reduction"""
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss(BaseLoss):
    """Focal Loss for classification"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma
        
    def __call__(self, pred, target):
        import torch
        import torch.nn.functional as F
        
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return self._reduce(focal_loss)


class LabelSmoothingLoss(BaseLoss):
    """Label Smoothing Loss"""
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__(reduction)
        self.smoothing = smoothing
        
    def __call__(self, pred, target):
        import torch
        import torch.nn.functional as F
        
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        # 创建平滑标签
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
            
        loss = (-smooth_target * log_preds).sum(dim=-1)
        return self._reduce(loss)


class DiceLoss(BaseLoss):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.smooth = smooth
        
    def __call__(self, pred, target):
        import torch
        
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1 - dice


class IoULoss(BaseLoss):
    """IoU Loss for detection"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)
        
    def __call__(self, pred_boxes, target_boxes):
        import torch
        
        # 计算交集
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算并集
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        loss = 1 - iou
        return self._reduce(loss)


# 注册自定义损失函数
LOSSES.register('FocalLoss', FocalLoss, category='classification', source=ComponentSource.BUILTIN)
LOSSES.register('LabelSmoothingLoss', LabelSmoothingLoss, category='classification', source=ComponentSource.BUILTIN)
LOSSES.register('DiceLoss', DiceLoss, category='segmentation', source=ComponentSource.BUILTIN)
LOSSES.register('IoULoss', IoULoss, category='detection', source=ComponentSource.BUILTIN)
