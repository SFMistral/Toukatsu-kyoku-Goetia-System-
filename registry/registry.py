# -*- coding: utf-8 -*-
"""
注册器基类模块

提供通用的组件注册与构建能力，支持装饰器注册、函数式注册、别名注册等。
"""

import time
import threading
import inspect
from typing import Dict, Any, Optional, List, Type, Callable, Union
from enum import Enum


class ConflictPolicy(Enum):
    """注册冲突处理策略"""
    ERROR = "error"      # 抛出异常
    WARN = "warn"        # 警告并覆盖
    OVERRIDE = "override"  # 静默覆盖
    SKIP = "skip"        # 静默跳过


class ComponentSource(Enum):
    """组件来源"""
    BUILTIN = "builtin"    # 内置组件
    USER = "user"          # 用户自定义
    THIRD_PARTY = "third_party"  # 第三方


class ComponentMeta:
    """组件元信息"""
    
    def __init__(
        self,
        cls: Type,
        name: str,
        aliases: Optional[List[str]] = None,
        category: Optional[str] = None,
        description: Optional[str] = None,
        source: ComponentSource = ComponentSource.USER,
        registered_at: Optional[float] = None
    ):
        self.cls = cls
        self.name = name
        self.aliases = aliases or []
        self.category = category
        self.description = description
        self.source = source
        self.registered_at = registered_at or time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'class': f"{self.cls.__module__}.{self.cls.__name__}",
            'aliases': self.aliases,
            'category': self.category,
            'description': self.description,
            'source': self.source.value,
            'registered_at': self.registered_at
        }


class Registry:
    """注册器基类"""
    
    def __init__(
        self,
        name: str,
        base_class: Optional[Type] = None,
        conflict_policy: ConflictPolicy = ConflictPolicy.ERROR
    ):
        """
        初始化注册器
        
        Args:
            name: 注册器名称
            base_class: 可选的基类约束，注册时会检查是否为其子类
            conflict_policy: 注册冲突处理策略
        """
        self.name = name
        self.base_class = base_class
        self.conflict_policy = conflict_policy
        
        # 组件映射表
        self._components: Dict[str, ComponentMeta] = {}
        self._alias_map: Dict[str, str] = {}  # alias -> canonical name
        
        # 线程安全
        self._lock = threading.RLock()

    def register(
        self,
        name: Optional[str] = None,
        cls: Optional[Type] = None,
        aliases: Optional[List[str]] = None,
        category: Optional[str] = None,
        description: Optional[str] = None,
        source: ComponentSource = ComponentSource.USER,
        force: bool = False
    ) -> Union[Type, Callable[[Type], Type]]:
        """
        注册组件
        
        支持装饰器和函数式两种用法:
        - 装饰器: @registry.register('name')
        - 函数式: registry.register('name', SomeClass)
        
        Args:
            name: 组件名称，为None时使用类名
            cls: 要注册的类（函数式调用时使用）
            aliases: 别名列表
            category: 组件分类
            description: 组件描述
            source: 组件来源
            force: 是否强制覆盖已存在的注册
            
        Returns:
            装饰器模式返回装饰器函数，函数式模式返回注册的类
        """
        # 函数式调用
        if cls is not None:
            self._do_register(
                cls=cls,
                name=name or cls.__name__,
                aliases=aliases,
                category=category,
                description=description,
                source=source,
                force=force
            )
            return cls
            
        # 装饰器调用
        def decorator(target_cls: Type) -> Type:
            self._do_register(
                cls=target_cls,
                name=name or target_cls.__name__,
                aliases=aliases,
                category=category,
                description=description,
                source=source,
                force=force
            )
            return target_cls
            
        return decorator
        
    def _do_register(
        self,
        cls: Type,
        name: str,
        aliases: Optional[List[str]] = None,
        category: Optional[str] = None,
        description: Optional[str] = None,
        source: ComponentSource = ComponentSource.USER,
        force: bool = False
    ):
        """执行实际的注册操作"""
        with self._lock:
            # 检查基类约束
            if self.base_class is not None:
                if not (inspect.isclass(cls) and issubclass(cls, self.base_class)):
                    raise TypeError(
                        f"Component '{name}' must be a subclass of {self.base_class.__name__}, "
                        f"got {cls}"
                    )
                    
            # 检查冲突
            if name in self._components and not force:
                self._handle_conflict(name, cls)
                if self.conflict_policy == ConflictPolicy.SKIP:
                    return
                    
            # 创建元信息
            meta = ComponentMeta(
                cls=cls,
                name=name,
                aliases=aliases,
                category=category,
                description=description,
                source=source
            )
            
            # 注册主名称
            self._components[name] = meta
            
            # 注册别名
            if aliases:
                for alias in aliases:
                    if alias in self._alias_map and not force:
                        self._handle_conflict(alias, cls, is_alias=True)
                        if self.conflict_policy == ConflictPolicy.SKIP:
                            continue
                    self._alias_map[alias] = name
                    
    def _handle_conflict(self, name: str, cls: Type, is_alias: bool = False):
        """处理注册冲突"""
        conflict_type = "Alias" if is_alias else "Component"
        existing = self._alias_map.get(name) if is_alias else name
        
        if self.conflict_policy == ConflictPolicy.ERROR:
            raise KeyError(
                f"{conflict_type} '{name}' is already registered in {self.name}"
            )
        elif self.conflict_policy == ConflictPolicy.WARN:
            import warnings
            warnings.warn(
                f"{conflict_type} '{name}' is already registered in {self.name}, "
                f"overriding with {cls.__name__}"
            )
        # OVERRIDE 和 SKIP 不需要额外处理
        
    def register_module(self, module, prefix: str = "", **kwargs):
        """
        批量注册模块内的所有类
        
        Args:
            module: Python模块对象
            prefix: 名称前缀
            **kwargs: 传递给register的其他参数
        """
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
                
            attr = getattr(module, attr_name)
            if not inspect.isclass(attr):
                continue
                
            # 检查基类约束
            if self.base_class is not None:
                if not issubclass(attr, self.base_class):
                    continue
                    
            name = f"{prefix}{attr_name}" if prefix else attr_name
            self.register(name=name, cls=attr, **kwargs)

    def get(self, name: str) -> Type:
        """
        获取注册的类
        
        Args:
            name: 组件名称或别名
            
        Returns:
            注册的类
            
        Raises:
            KeyError: 组件未注册
        """
        with self._lock:
            # 先查找主名称
            if name in self._components:
                return self._components[name].cls
                
            # 再查找别名
            if name in self._alias_map:
                canonical_name = self._alias_map[name]
                return self._components[canonical_name].cls
                
            raise KeyError(
                f"Component '{name}' is not registered in {self.name}. "
                f"Available: {list(self._components.keys())}"
            )
            
    def get_meta(self, name: str) -> ComponentMeta:
        """获取组件元信息"""
        with self._lock:
            if name in self._components:
                return self._components[name]
                
            if name in self._alias_map:
                canonical_name = self._alias_map[name]
                return self._components[canonical_name]
                
            raise KeyError(f"Component '{name}' is not registered in {self.name}")
            
    def contains(self, name: str) -> bool:
        """检查组件是否已注册"""
        with self._lock:
            return name in self._components or name in self._alias_map
            
    def __contains__(self, name: str) -> bool:
        """支持 'name' in registry 语法"""
        return self.contains(name)
        
    def list_all(self) -> List[str]:
        """列出所有已注册组件名称"""
        with self._lock:
            return list(self._components.keys())
            
    def list_by_category(self, category: str) -> List[str]:
        """按类别筛选组件"""
        with self._lock:
            return [
                name for name, meta in self._components.items()
                if meta.category == category
            ]
            
    def list_by_source(self, source: ComponentSource) -> List[str]:
        """按来源筛选组件"""
        with self._lock:
            return [
                name for name, meta in self._components.items()
                if meta.source == source
            ]
            
    def get_all_meta(self) -> Dict[str, ComponentMeta]:
        """获取所有组件的元信息"""
        with self._lock:
            return dict(self._components)
            
    def build(
        self,
        config: Dict[str, Any],
        default_args: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        根据配置构建组件实例
        
        Args:
            config: 配置字典，必须包含'type'字段指定组件名称
            default_args: 默认参数，会被config中的参数覆盖
            
        Returns:
            构建的组件实例
        """
        if not isinstance(config, dict):
            raise TypeError(f"Config must be a dict, got {type(config)}")
            
        config = config.copy()
        
        # 获取组件类型
        component_type = config.pop('type', None)
        if component_type is None:
            raise KeyError("Config must contain 'type' field")
            
        # 获取组件类
        cls = self.get(component_type)
        
        # 合并参数
        if default_args:
            for key, value in default_args.items():
                config.setdefault(key, value)
                
        # 递归构建嵌套配置
        config = self._build_nested(config)
        
        # 实例化
        return cls(**config)
        
    def _build_nested(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """递归构建嵌套配置中的组件"""
        result = {}
        for key, value in config.items():
            if isinstance(value, dict) and 'type' in value:
                # 尝试从当前注册器构建
                try:
                    result[key] = self.build(value)
                except KeyError:
                    # 如果当前注册器没有，保持原样
                    result[key] = value
            elif isinstance(value, list):
                result[key] = [
                    self.build(item) if isinstance(item, dict) and 'type' in item else item
                    for item in value
                ]
            else:
                result[key] = value
        return result
        
    def build_many(
        self,
        configs: List[Dict[str, Any]],
        default_args: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """批量构建组件"""
        return [self.build(cfg, default_args) for cfg in configs]
        
    def unregister(self, name: str):
        """取消注册组件"""
        with self._lock:
            if name in self._components:
                meta = self._components.pop(name)
                # 移除相关别名
                for alias in meta.aliases:
                    self._alias_map.pop(alias, None)
            elif name in self._alias_map:
                self._alias_map.pop(name)
            else:
                raise KeyError(f"Component '{name}' is not registered")
                
    def clear(self):
        """清空所有注册"""
        with self._lock:
            self._components.clear()
            self._alias_map.clear()
            
    def __len__(self) -> int:
        """返回注册组件数量"""
        return len(self._components)
        
    def __repr__(self) -> str:
        return f"Registry(name='{self.name}', components={len(self._components)})"


def build_from_config(config: Dict[str, Any], registry: Registry, **kwargs) -> Any:
    """
    通用构建函数
    
    Args:
        config: 配置字典
        registry: 注册器实例
        **kwargs: 传递给build的额外参数
        
    Returns:
        构建的组件实例
    """
    return registry.build(config, **kwargs)
