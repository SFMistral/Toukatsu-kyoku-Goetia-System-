# -*- coding: utf-8 -*-
"""
数据增强注册器模块

管理数据增强相关组件：单样本变换、批级变换等。
支持Pipeline组合构建和参数校验。
"""

from typing import Dict, Any, Optional, List, Type, Callable, Union
from .registry import Registry, ComponentSource


class AugmentationRegistry(Registry):
    """数据增强注册器，支持Pipeline构建"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._param_ranges: Dict[str, Dict[str, tuple]] = {}
        
    def register(
        self,
        name: Optional[str] = None,
        cls: Optional[Type] = None,
        param_ranges: Optional[Dict[str, tuple]] = None,
        **kwargs
    ):
        """
        注册数据增强组件
        
        Args:
            name: 组件名称
            cls: 组件类
            param_ranges: 参数有效范围，如 {'p': (0.0, 1.0), 'scale': (0.5, 2.0)}
            **kwargs: 其他注册参数
        """
        result = super().register(name=name, cls=cls, **kwargs)
        
        actual_name = name or (cls.__name__ if cls else None)
        if actual_name and param_ranges:
            self._param_ranges[actual_name] = param_ranges
            
        return result
        
    def validate_params(self, name: str, params: Dict[str, Any]) -> bool:
        """
        校验变换参数是否在有效范围内
        
        Args:
            name: 变换名称
            params: 参数字典
            
        Returns:
            是否有效
        """
        if name not in self._param_ranges:
            return True
            
        ranges = self._param_ranges[name]
        for param_name, value in params.items():
            if param_name in ranges:
                min_val, max_val = ranges[param_name]
                if not (min_val <= value <= max_val):
                    return False
        return True
        
    def build_pipeline(self, configs: List[Dict[str, Any]]) -> 'TransformPipeline':
        """
        构建变换Pipeline
        
        Args:
            configs: 变换配置列表
            
        Returns:
            TransformPipeline实例
        """
        transforms = [self.build(cfg) for cfg in configs]
        return TransformPipeline(transforms)


class TransformPipeline:
    """变换Pipeline，按顺序应用多个变换"""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
        
    def __call__(self, data):
        """应用所有变换"""
        for transform in self.transforms:
            data = transform(data)
        return data
        
    def __repr__(self):
        transform_names = [t.__class__.__name__ for t in self.transforms]
        return f"TransformPipeline({transform_names})"


# 创建数据增强相关注册器单例
TRANSFORMS = AugmentationRegistry('transforms', base_class=None)
BATCH_TRANSFORMS = AugmentationRegistry('batch_transforms', base_class=None)


class BaseTransform:
    """变换基类"""
    
    def __init__(self, p: float = 1.0):
        """
        Args:
            p: 应用变换的概率
        """
        self.p = p
        
    def __call__(self, data):
        import random
        if random.random() < self.p:
            return self.apply(data)
        return data
        
    def apply(self, data):
        """实际的变换逻辑，子类实现"""
        raise NotImplementedError


# 几何变换
class Resize(BaseTransform):
    """调整大小"""
    
    def __init__(self, size: Union[int, tuple], interpolation: str = 'bilinear', **kwargs):
        super().__init__(**kwargs)
        self.size = size if isinstance(size, tuple) else (size, size)
        self.interpolation = interpolation
        
    def apply(self, data):
        import torch.nn.functional as F
        
        if isinstance(data, dict):
            image = data['image']
            resized = F.interpolate(
                image.unsqueeze(0), 
                size=self.size, 
                mode=self.interpolation
            ).squeeze(0)
            data['image'] = resized
            return data
        return F.interpolate(data.unsqueeze(0), size=self.size, mode=self.interpolation).squeeze(0)


class RandomCrop(BaseTransform):
    """随机裁剪"""
    
    def __init__(self, size: Union[int, tuple], **kwargs):
        super().__init__(**kwargs)
        self.size = size if isinstance(size, tuple) else (size, size)
        
    def apply(self, data):
        import random
        
        if isinstance(data, dict):
            image = data['image']
        else:
            image = data
            
        _, h, w = image.shape
        th, tw = self.size
        
        if h < th or w < tw:
            return data
            
        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)
        
        cropped = image[:, top:top+th, left:left+tw]
        
        if isinstance(data, dict):
            data['image'] = cropped
            return data
        return cropped


class RandomFlip(BaseTransform):
    """随机翻转"""
    
    def __init__(self, horizontal: bool = True, vertical: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.horizontal = horizontal
        self.vertical = vertical
        
    def apply(self, data):
        import random
        import torch
        
        if isinstance(data, dict):
            image = data['image']
        else:
            image = data
            
        if self.horizontal and random.random() < 0.5:
            image = torch.flip(image, dims=[-1])
            
        if self.vertical and random.random() < 0.5:
            image = torch.flip(image, dims=[-2])
            
        if isinstance(data, dict):
            data['image'] = image
            return data
        return image


class RandomRotate(BaseTransform):
    """随机旋转"""
    
    def __init__(self, degrees: float = 15, **kwargs):
        super().__init__(**kwargs)
        self.degrees = degrees
        
    def apply(self, data):
        import random
        import torch
        import torch.nn.functional as F
        import math
        
        if isinstance(data, dict):
            image = data['image']
        else:
            image = data
            
        angle = random.uniform(-self.degrees, self.degrees)
        angle_rad = math.radians(angle)
        
        # 简化的旋转实现
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=image.dtype, device=image.device).unsqueeze(0)
        
        grid = F.affine_grid(theta, image.unsqueeze(0).shape, align_corners=False)
        rotated = F.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0)
        
        if isinstance(data, dict):
            data['image'] = rotated
            return data
        return rotated


# 颜色变换
class Normalize(BaseTransform):
    """归一化"""
    
    def __init__(self, mean: List[float], std: List[float], **kwargs):
        super().__init__(p=1.0)  # 归一化总是应用
        self.mean = mean
        self.std = std
        
    def apply(self, data):
        import torch
        
        if isinstance(data, dict):
            image = data['image']
        else:
            image = data
            
        mean = torch.tensor(self.mean, device=image.device).view(-1, 1, 1)
        std = torch.tensor(self.std, device=image.device).view(-1, 1, 1)
        
        normalized = (image - mean) / std
        
        if isinstance(data, dict):
            data['image'] = normalized
            return data
        return normalized


class ColorJitter(BaseTransform):
    """颜色抖动"""
    
    def __init__(
        self, 
        brightness: float = 0.2, 
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
    def apply(self, data):
        import random
        import torch
        
        if isinstance(data, dict):
            image = data['image']
        else:
            image = data
            
        # 亮度调整
        if self.brightness > 0:
            factor = 1 + random.uniform(-self.brightness, self.brightness)
            image = image * factor
            
        # 对比度调整
        if self.contrast > 0:
            factor = 1 + random.uniform(-self.contrast, self.contrast)
            mean = image.mean()
            image = (image - mean) * factor + mean
            
        image = torch.clamp(image, 0, 1)
        
        if isinstance(data, dict):
            data['image'] = image
            return data
        return image


# 批级变换
class Mixup(BaseTransform):
    """Mixup数据增强"""
    
    def __init__(self, alpha: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        
    def apply(self, data):
        import torch
        import random
        
        images, labels = data['images'], data['labels']
        batch_size = images.size(0)
        
        # 生成混合系数
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        
        # 随机打乱索引
        indices = torch.randperm(batch_size)
        
        # 混合图像
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        return {
            'images': mixed_images,
            'labels': labels,
            'labels_aux': labels[indices],
            'lam': lam
        }


class CutMix(BaseTransform):
    """CutMix数据增强"""
    
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        
    def apply(self, data):
        import torch
        import random
        
        images, labels = data['images'], data['labels']
        batch_size, _, h, w = images.shape
        
        # 生成混合系数
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        
        # 随机打乱索引
        indices = torch.randperm(batch_size)
        
        # 计算裁剪区域
        cut_ratio = (1 - lam).sqrt()
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        
        x1 = max(0, cx - cut_w // 2)
        x2 = min(w, cx + cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(h, cy + cut_h // 2)
        
        # 混合图像
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
        
        # 调整lambda
        lam = 1 - (x2 - x1) * (y2 - y1) / (h * w)
        
        return {
            'images': mixed_images,
            'labels': labels,
            'labels_aux': labels[indices],
            'lam': lam
        }


# 注册内置变换
TRANSFORMS.register('Resize', Resize, category='geometric', 
                   param_ranges={'p': (0.0, 1.0)}, source=ComponentSource.BUILTIN)
TRANSFORMS.register('RandomCrop', RandomCrop, category='geometric',
                   param_ranges={'p': (0.0, 1.0)}, source=ComponentSource.BUILTIN)
TRANSFORMS.register('RandomFlip', RandomFlip, category='geometric',
                   param_ranges={'p': (0.0, 1.0)}, source=ComponentSource.BUILTIN)
TRANSFORMS.register('RandomRotate', RandomRotate, category='geometric',
                   param_ranges={'p': (0.0, 1.0), 'degrees': (0, 180)}, source=ComponentSource.BUILTIN)
TRANSFORMS.register('Normalize', Normalize, category='color', source=ComponentSource.BUILTIN)
TRANSFORMS.register('ColorJitter', ColorJitter, category='color',
                   param_ranges={'brightness': (0.0, 1.0), 'contrast': (0.0, 1.0)}, 
                   source=ComponentSource.BUILTIN)

BATCH_TRANSFORMS.register('Mixup', Mixup, category='batch',
                         param_ranges={'alpha': (0.0, 2.0)}, source=ComponentSource.BUILTIN)
BATCH_TRANSFORMS.register('CutMix', CutMix, category='batch',
                         param_ranges={'alpha': (0.0, 2.0)}, source=ComponentSource.BUILTIN)


def _register_torchvision_transforms():
    """从torchvision导入变换"""
    try:
        from torchvision import transforms as T
        
        tv_transforms = [
            ('ToTensor', T.ToTensor),
            ('ToPILImage', T.ToPILImage),
            ('CenterCrop', T.CenterCrop),
            ('RandomResizedCrop', T.RandomResizedCrop),
            ('RandomHorizontalFlip', T.RandomHorizontalFlip),
            ('RandomVerticalFlip', T.RandomVerticalFlip),
            ('RandomGrayscale', T.RandomGrayscale),
            ('GaussianBlur', T.GaussianBlur),
        ]
        
        for name, cls in tv_transforms:
            if name not in TRANSFORMS:
                TRANSFORMS.register(
                    name=f"tv_{name}",
                    cls=cls,
                    category='torchvision',
                    source=ComponentSource.THIRD_PARTY
                )
    except ImportError:
        pass
