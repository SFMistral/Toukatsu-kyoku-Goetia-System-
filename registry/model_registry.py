# -*- coding: utf-8 -*-
"""
模型注册器模块

管理模型相关组件：完整模型、骨干网络、特征融合层、任务头等。
"""

from typing import Dict, Any, Optional, Type
from .registry import Registry, ComponentSource


class ModelRegistry(Registry):
    """模型注册器，支持预训练权重自动加载"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._pretrained_urls: Dict[str, str] = {}
        
    def register(
        self,
        name: Optional[str] = None,
        cls: Optional[Type] = None,
        pretrained_url: Optional[str] = None,
        **kwargs
    ):
        """
        注册模型组件
        
        Args:
            name: 组件名称
            cls: 组件类
            pretrained_url: 预训练权重URL
            **kwargs: 其他注册参数
        """
        result = super().register(name=name, cls=cls, **kwargs)
        
        # 记录预训练权重URL
        if pretrained_url and name:
            self._pretrained_urls[name] = pretrained_url
            
        return result
        
    def build(
        self,
        config: Dict[str, Any],
        default_args: Optional[Dict[str, Any]] = None,
        load_pretrained: bool = True
    ) -> Any:
        """
        构建模型实例
        
        Args:
            config: 模型配置
            default_args: 默认参数
            load_pretrained: 是否加载预训练权重
        """
        config = config.copy()
        pretrained = config.pop('pretrained', None)
        
        # 构建模型
        model = super().build(config, default_args)
        
        # 加载预训练权重
        if load_pretrained and pretrained:
            self._load_pretrained(model, pretrained, config.get('type'))
            
        return model
        
    def _load_pretrained(self, model, pretrained, model_type: Optional[str] = None):
        """加载预训练权重"""
        try:
            import torch
            
            if isinstance(pretrained, bool) and pretrained:
                # 使用注册的预训练URL
                if model_type and model_type in self._pretrained_urls:
                    url = self._pretrained_urls[model_type]
                    state_dict = torch.hub.load_state_dict_from_url(url)
                    model.load_state_dict(state_dict, strict=False)
            elif isinstance(pretrained, str):
                # 从路径或URL加载
                if pretrained.startswith(('http://', 'https://')):
                    state_dict = torch.hub.load_state_dict_from_url(pretrained)
                else:
                    state_dict = torch.load(pretrained, map_location='cpu')
                    
                # 处理可能的嵌套结构
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                    
                model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load pretrained weights: {e}")
            
    def get_pretrained_url(self, name: str) -> Optional[str]:
        """获取预训练权重URL"""
        return self._pretrained_urls.get(name)


# 创建模型相关注册器单例
MODELS = ModelRegistry('models', base_class=None)
BACKBONES = ModelRegistry('backbones', base_class=None)
NECKS = ModelRegistry('necks', base_class=None)
HEADS = ModelRegistry('heads', base_class=None)


def _register_builtin_models():
    """注册内置模型（延迟导入）"""
    pass  # 实际模型在具体模块中注册


def _register_timm_models():
    """从timm自动导入模型"""
    try:
        import timm
        
        class TimmModelWrapper:
            """timm模型包装器"""
            def __init__(self, model_name: str, **kwargs):
                self.model = timm.create_model(model_name, **kwargs)
                
            def forward(self, x):
                return self.model(x)
                
            def __getattr__(self, name):
                return getattr(self.model, name)
                
        # 注册常用timm模型
        timm_models = [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
            'vit_base_patch16_224', 'vit_small_patch16_224',
            'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224',
        ]
        
        for model_name in timm_models:
            if model_name not in BACKBONES:
                # 创建工厂函数
                def make_factory(name):
                    def factory(**kwargs):
                        return timm.create_model(name, **kwargs)
                    factory.__name__ = f"timm_{name}"
                    return factory
                    
                BACKBONES.register(
                    name=f"timm_{model_name}",
                    cls=make_factory(model_name),
                    category='timm',
                    source=ComponentSource.THIRD_PARTY
                )
    except ImportError:
        pass  # timm未安装


def _register_torchvision_models():
    """从torchvision自动导入模型"""
    try:
        from torchvision import models as tv_models
        
        torchvision_backbones = [
            ('resnet18', tv_models.resnet18),
            ('resnet34', tv_models.resnet34),
            ('resnet50', tv_models.resnet50),
            ('resnet101', tv_models.resnet101),
            ('vgg16', tv_models.vgg16),
            ('vgg19', tv_models.vgg19),
        ]
        
        for name, model_fn in torchvision_backbones:
            if f"tv_{name}" not in BACKBONES:
                BACKBONES.register(
                    name=f"tv_{name}",
                    cls=model_fn,
                    category='torchvision',
                    source=ComponentSource.THIRD_PARTY
                )
    except ImportError:
        pass  # torchvision未安装
