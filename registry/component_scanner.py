# -*- coding: utf-8 -*-
"""
组件自动扫描器模块

自动扫描并注册指定路径下的组件。
"""

import os
import sys
import importlib
import importlib.util
import fnmatch
import time
from typing import Dict, Any, Optional, List, Set, Type
from enum import Enum
from dataclasses import dataclass, field

from .registry import Registry, ComponentSource


class ScanMode(Enum):
    """扫描模式"""
    EAGER = "eager"    # 启动时立即扫描
    LAZY = "lazy"      # 首次访问时扫描
    MANUAL = "manual"  # 仅手动触发


@dataclass
class ScanResult:
    """扫描结果"""
    discovered: List[str] = field(default_factory=list)
    registered: List[str] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    scan_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'discovered': self.discovered,
            'registered': self.registered,
            'conflicts': self.conflicts,
            'errors': self.errors,
            'scan_time': self.scan_time
        }
        
    def summary(self) -> str:
        return (
            f"Scan completed in {self.scan_time:.2f}s: "
            f"{len(self.discovered)} discovered, "
            f"{len(self.registered)} registered, "
            f"{len(self.conflicts)} conflicts, "
            f"{len(self.errors)} errors"
        )


class ComponentScanner:
    """组件自动扫描器"""
    
    # 默认扫描路径映射
    DEFAULT_SCAN_PATHS = {
        'models': ['models/', 'backbones/', 'necks/', 'heads/'],
        'datasets': ['datasets/', 'data/'],
        'losses': ['losses/'],
        'metrics': ['metrics/'],
        'transforms': ['transforms/', 'augmentations/'],
        'optimizers': ['optimizers/'],
        'schedulers': ['schedulers/'],
        'hooks': ['hooks/'],
        'exporters': ['exporters/'],
    }
    
    # 默认排除模式
    DEFAULT_EXCLUDE_PATTERNS = [
        '__pycache__',
        '*.pyc',
        '__init__.py',
        'test_*.py',
        '*_test.py',
        'conftest.py',
        'setup.py',
    ]
    
    def __init__(
        self,
        scan_paths: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        base_classes: Optional[Dict[str, Type]] = None,
        registries: Optional[Dict[str, Registry]] = None,
        auto_register: bool = True,
        scan_mode: ScanMode = ScanMode.EAGER
    ):
        """
        初始化扫描器
        
        Args:
            scan_paths: 扫描路径列表
            exclude_patterns: 排除模式列表
            base_classes: 基类映射 {category: base_class}
            registries: 注册器映射 {category: registry}
            auto_register: 是否自动注册发现的组件
            scan_mode: 扫描模式
        """
        self.scan_paths = scan_paths or []
        self.exclude_patterns = exclude_patterns or self.DEFAULT_EXCLUDE_PATTERNS
        self.base_classes = base_classes or {}
        self.registries = registries or {}
        self.auto_register = auto_register
        self.scan_mode = scan_mode
        
        self._scanned = False
        self._result = ScanResult()
        self._discovered_components: Dict[str, Type] = {}
        
    def scan(self, force: bool = False) -> ScanResult:
        """
        执行扫描
        
        Args:
            force: 是否强制重新扫描
            
        Returns:
            扫描结果
        """
        if self._scanned and not force:
            return self._result
            
        start_time = time.time()
        self._result = ScanResult()
        self._discovered_components.clear()
        
        for path in self.scan_paths:
            self._scan_path(path)
            
        self._result.scan_time = time.time() - start_time
        self._scanned = True
        
        return self._result
        
    def _scan_path(self, path: str):
        """扫描单个路径"""
        if not os.path.exists(path):
            self._result.errors.append({
                'path': path,
                'error': 'Path does not exist'
            })
            return
            
        if os.path.isfile(path):
            if path.endswith('.py'):
                self._scan_file(path)
        else:
            for root, dirs, files in os.walk(path):
                # 过滤排除的目录
                dirs[:] = [d for d in dirs if not self._should_exclude(d)]
                
                for file in files:
                    if file.endswith('.py') and not self._should_exclude(file):
                        filepath = os.path.join(root, file)
                        self._scan_file(filepath)
                        
    def _should_exclude(self, name: str) -> bool:
        """检查是否应该排除"""
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False
        
    def _scan_file(self, filepath: str):
        """扫描单个文件"""
        try:
            module = self._load_module(filepath)
            if module is None:
                return
                
            self._scan_module(module, filepath)
            
        except Exception as e:
            self._result.errors.append({
                'path': filepath,
                'error': str(e)
            })
            
    def _load_module(self, filepath: str):
        """动态加载模块"""
        try:
            # 生成模块名
            module_name = os.path.splitext(filepath)[0].replace(os.sep, '.')
            module_name = module_name.lstrip('.')
            
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec is None or spec.loader is None:
                return None
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            return module
            
        except Exception as e:
            self._result.errors.append({
                'path': filepath,
                'error': f'Failed to load module: {e}'
            })
            return None
            
    def _scan_module(self, module, filepath: str):
        """扫描模块中的组件"""
        import inspect
        
        for name, obj in inspect.getmembers(module):
            # 跳过私有成员
            if name.startswith('_'):
                continue
                
            # 跳过导入的成员
            if hasattr(obj, '__module__') and obj.__module__ != module.__name__:
                continue
                
            # 检查是否是类
            if not inspect.isclass(obj):
                continue
                
            # 检查是否有注册标记
            if self._should_register(obj):
                self._result.discovered.append(f"{module.__name__}.{name}")
                self._discovered_components[name] = obj
                
                if self.auto_register:
                    self._register_component(name, obj)
                    
    def _should_register(self, cls: Type) -> bool:
        """判断类是否应该被注册"""
        # 检查是否有 @register 装饰器标记
        if hasattr(cls, '_registry_marked'):
            return True
            
        # 检查是否继承自指定基类
        for base_class in self.base_classes.values():
            if base_class is not None and issubclass(cls, base_class):
                return True
                
        # 检查是否是 nn.Module 子类
        try:
            import torch.nn as nn
            if issubclass(cls, nn.Module):
                return True
        except ImportError:
            pass
            
        return False
        
    def _register_component(self, name: str, cls: Type):
        """注册组件到对应的注册器"""
        # 确定组件类别
        category = self._determine_category(cls)
        
        if category and category in self.registries:
            registry = self.registries[category]
            
            try:
                if registry.contains(name):
                    self._result.conflicts.append({
                        'name': name,
                        'class': f"{cls.__module__}.{cls.__name__}",
                        'registry': registry.name
                    })
                else:
                    registry.register(
                        name=name,
                        cls=cls,
                        source=ComponentSource.USER
                    )
                    self._result.registered.append(name)
            except Exception as e:
                self._result.errors.append({
                    'name': name,
                    'error': str(e)
                })
                
    def _determine_category(self, cls: Type) -> Optional[str]:
        """确定组件类别"""
        # 根据基类判断
        for category, base_class in self.base_classes.items():
            if base_class is not None and issubclass(cls, base_class):
                return category
                
        # 根据类名后缀判断
        class_name = cls.__name__.lower()
        
        if class_name.endswith('loss'):
            return 'losses'
        elif class_name.endswith('metric'):
            return 'metrics'
        elif class_name.endswith('dataset'):
            return 'datasets'
        elif class_name.endswith('transform'):
            return 'transforms'
        elif class_name.endswith('hook'):
            return 'hooks'
        elif class_name.endswith('exporter'):
            return 'exporters'
        elif class_name.endswith('scheduler'):
            return 'schedulers'
        elif class_name.endswith('optimizer'):
            return 'optimizers'
            
        # 默认为模型
        return 'models'
        
    def get_report(self) -> ScanResult:
        """获取扫描报告"""
        return self._result
        
    def get_discovered(self) -> Dict[str, Type]:
        """获取发现的组件"""
        return dict(self._discovered_components)
        
    def rescan(self) -> ScanResult:
        """重新扫描"""
        return self.scan(force=True)
        
    def add_path(self, path: str):
        """添加扫描路径"""
        if path not in self.scan_paths:
            self.scan_paths.append(path)
            
    def remove_path(self, path: str):
        """移除扫描路径"""
        if path in self.scan_paths:
            self.scan_paths.remove(path)
            
    def set_registry(self, category: str, registry: Registry):
        """设置类别对应的注册器"""
        self.registries[category] = registry
        
    def set_base_class(self, category: str, base_class: Type):
        """设置类别对应的基类"""
        self.base_classes[category] = base_class


def register_marker(cls: Type) -> Type:
    """
    标记类为可注册组件的装饰器
    
    Usage:
        @register_marker
        class MyModel:
            pass
    """
    cls._registry_marked = True
    return cls


def create_scanner_from_config(config: Dict[str, Any]) -> ComponentScanner:
    """
    从配置创建扫描器
    
    Args:
        config: 扫描器配置
        
    Returns:
        ComponentScanner实例
    """
    scan_mode = ScanMode(config.get('scan_mode', 'eager'))
    
    scanner = ComponentScanner(
        scan_paths=config.get('scan_paths', []),
        exclude_patterns=config.get('exclude_patterns'),
        auto_register=config.get('auto_register', True),
        scan_mode=scan_mode
    )
    
    # 如果是eager模式，立即扫描
    if scan_mode == ScanMode.EAGER:
        scanner.scan()
        
    return scanner
