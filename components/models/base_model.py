# -*- coding: utf-8 -*-
"""
模型基类模块

定义所有模型的抽象基类，提供统一接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.nn as nn

from .utils import init_weights, count_parameters, load_checkpoint


class BaseModel(nn.Module, ABC):
    """
    所有模型的抽象基类
    
    定义模型通用接口，提供前向传播框架，统一训练/推理模式切换。
    
    Args:
        init_cfg: 权重初始化配置
    """
    
    def __init__(self, init_cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.init_cfg = init_cfg
        self._is_init = False
        
    def init_weights(self):
        """初始化权重"""
        if self.init_cfg is not None:
            init_weights(self, self.init_cfg)
        self._is_init = True
        
    def load_pretrained(self, path: str, strict: bool = False) -> Dict[str, Any]:
        """
        加载预训练权重
        
        Args:
            path: 权重文件路径或URL
            strict: 是否严格匹配
            
        Returns:
            加载信息（missing_keys, unexpected_keys）
        """
        return load_checkpoint(self, path, strict=strict)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[Any] = None,
        mode: str = 'loss',
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """
        统一前向接口
        
        Args:
            inputs: 输入张量
            targets: 目标标签（训练时需要）
            mode: 模式
                - 'loss': 训练模式，返回损失字典
                - 'predict': 推理模式，返回预测结果
                - 'features': 特征模式，返回特征图
                
        Returns:
            根据mode返回不同结果
        """
        if mode == 'loss':
            return self.forward_train(inputs, targets)
        elif mode == 'predict':
            return self.forward_test(inputs)
        elif mode == 'features':
            return self.forward_features(inputs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
    @abstractmethod
    def forward_train(
        self,
        inputs: torch.Tensor,
        targets: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        训练前向传播
        
        Args:
            inputs: 输入张量
            targets: 目标标签
            
        Returns:
            损失字典 {'loss': tensor, ...}
        """
        pass
        
    @abstractmethod
    def forward_test(self, inputs: torch.Tensor) -> Any:
        """
        推理前向传播
        
        Args:
            inputs: 输入张量
            
        Returns:
            预测结果
        """
        pass
        
    def forward_features(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """
        特征提取
        
        Args:
            inputs: 输入张量
            
        Returns:
            特征图列表
        """
        raise NotImplementedError("forward_features not implemented")
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        total_params = count_parameters(self, trainable_only=False)
        trainable_params = count_parameters(self, trainable_only=True)
        
        return {
            'name': self.__class__.__name__,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params,
            'is_initialized': self._is_init,
        }
        
    def __repr__(self) -> str:
        info = self.get_model_info()
        return (
            f"{info['name']}("
            f"params={info['total_params']:,}, "
            f"trainable={info['trainable_params']:,})"
        )
