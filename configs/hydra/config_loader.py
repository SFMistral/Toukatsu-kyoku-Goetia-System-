"""
配置文件加载器

支持从 YAML、JSON、Python 字典加载配置，
支持 _base_ 字段实现配置继承。
"""

import json
import os
from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """配置加载器，支持多种来源和配置继承"""
    
    def __init__(self, search_paths: list[str] | None = None):
        """
        初始化配置加载器
        
        Args:
            search_paths: 配置文件搜索路径列表
        """
        self._cache: dict[str, dict] = {}
        self._loading_stack: list[str] = []  # 用于检测循环引用
        
        # 默认搜索路径
        self._search_paths = search_paths or [
            str(Path(__file__).parent.parent),  # configs/
            str(Path(__file__).parent.parent / "defaults"),
            str(Path(__file__).parent.parent / "templates"),
            str(Path(__file__).parent.parent / "groups"),
        ]
    
    def load(self, path: str, use_cache: bool = True) -> dict:
        """
        从文件加载配置
        
        Args:
            path: 配置文件路径（相对或绝对）
            use_cache: 是否使用缓存
        
        Returns:
            配置字典
        """
        resolved_path = self._resolve_path(path)
        
        if use_cache and resolved_path in self._cache:
            return self._cache[resolved_path].copy()
        
        # 循环引用检测
        if resolved_path in self._loading_stack:
            cycle = " -> ".join(self._loading_stack + [resolved_path])
            raise ValueError(f"检测到循环引用: {cycle}")
        
        self._loading_stack.append(resolved_path)
        
        try:
            config = self._load_file(resolved_path)
            
            # 处理 _base_ 继承
            if "_base_" in config:
                base_paths = config.pop("_base_")
                if isinstance(base_paths, str):
                    base_paths = [base_paths]
                
                base_config = {}
                base_dir = os.path.dirname(resolved_path)
                
                for base_path in base_paths:
                    # 相对于当前配置文件解析路径
                    if not os.path.isabs(base_path):
                        base_path = os.path.join(base_dir, base_path)
                    
                    parent_config = self.load(base_path, use_cache)
                    base_config = self._deep_merge(base_config, parent_config)
                
                config = self._deep_merge(base_config, config)
            
            if use_cache:
                self._cache[resolved_path] = config.copy()
            
            return config
            
        finally:
            self._loading_stack.pop()
    
    def load_from_dict(self, data: dict) -> dict:
        """
        从 Python 字典加载配置
        
        Args:
            data: 配置字典
        
        Returns:
            处理后的配置字典
        """
        return data.copy()
    
    def load_from_json(self, json_str: str) -> dict:
        """
        从 JSON 字符串加载配置
        
        Args:
            json_str: JSON 字符串
        
        Returns:
            配置字典
        """
        return json.loads(json_str)
    
    def clear_cache(self) -> None:
        """清除配置缓存"""
        self._cache.clear()
    
    def _resolve_path(self, path: str) -> str:
        """解析配置文件路径"""
        if os.path.isabs(path):
            if os.path.exists(path):
                return path
            raise FileNotFoundError(f"配置文件不存在: {path}")
        
        # 在搜索路径中查找
        for search_path in self._search_paths:
            full_path = os.path.join(search_path, path)
            if os.path.exists(full_path):
                return full_path
        
        # 尝试当前目录
        if os.path.exists(path):
            return os.path.abspath(path)
        
        raise FileNotFoundError(
            f"配置文件不存在: {path}\n"
            f"搜索路径: {self._search_paths}"
        )
    
    def _load_file(self, path: str) -> dict:
        """加载单个配置文件"""
        # 自动检测编码
        encodings = ["utf-8", "utf-8-sig", "gbk", "latin-1"]
        content = None
        
        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise ValueError(f"无法解码配置文件: {path}")
        
        # 根据扩展名解析
        ext = os.path.splitext(path)[1].lower()
        
        if ext in (".yaml", ".yml"):
            return yaml.safe_load(content) or {}
        elif ext == ".json":
            return json.loads(content)
        else:
            # 尝试 YAML
            try:
                return yaml.safe_load(content) or {}
            except yaml.YAMLError:
                return json.loads(content)
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """深度合并两个字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
