"""
配置组合器

实现多配置源的深度合并，支持配置组的组合。
"""

from enum import Enum
from typing import Any


class ListMergeStrategy(Enum):
    """列表合并策略"""
    REPLACE = "replace"  # 替换
    APPEND = "append"    # 追加
    PREPEND = "prepend"  # 前置
    MERGE = "merge"      # 按索引合并


class ConfigComposer:
    """配置组合器，实现多配置源的合并"""
    
    def __init__(
        self,
        list_strategy: ListMergeStrategy = ListMergeStrategy.REPLACE,
        preserve_none: bool = False,
    ):
        """
        初始化配置组合器
        
        Args:
            list_strategy: 列表合并策略
            preserve_none: 是否保留 None 值
        """
        self._list_strategy = list_strategy
        self._preserve_none = preserve_none
    
    def merge(self, base: dict, *others: dict) -> dict:
        """
        合并多个配置字典
        
        Args:
            base: 基础配置
            *others: 其他配置
        
        Returns:
            合并后的配置字典
        """
        result = self._deep_copy(base)
        
        for other in others:
            result = self._merge_two(result, other)
        
        return result
    
    def compose_groups(self, base: dict, groups: list[dict]) -> dict:
        """
        组合配置组
        
        Args:
            base: 基础配置
            groups: 配置组列表
        
        Returns:
            组合后的配置字典
        """
        return self.merge(base, *groups)
    
    def _merge_two(self, base: dict, override: dict) -> dict:
        """合并两个字典"""
        result = self._deep_copy(base)
        
        for key, value in override.items():
            if value is None and not self._preserve_none:
                continue
            
            if key in result:
                result[key] = self._merge_values(result[key], value)
            else:
                result[key] = self._deep_copy(value)
        
        return result
    
    def _merge_values(self, base: Any, override: Any) -> Any:
        """合并两个值"""
        # 字典递归合并
        if isinstance(base, dict) and isinstance(override, dict):
            return self._merge_two(base, override)
        
        # 列表按策略合并
        if isinstance(base, list) and isinstance(override, list):
            return self._merge_lists(base, override)
        
        # 标量直接覆盖
        return self._deep_copy(override)
    
    def _merge_lists(self, base: list, override: list) -> list:
        """按策略合并列表"""
        if self._list_strategy == ListMergeStrategy.REPLACE:
            return self._deep_copy(override)
        
        elif self._list_strategy == ListMergeStrategy.APPEND:
            return self._deep_copy(base) + self._deep_copy(override)
        
        elif self._list_strategy == ListMergeStrategy.PREPEND:
            return self._deep_copy(override) + self._deep_copy(base)
        
        elif self._list_strategy == ListMergeStrategy.MERGE:
            result = self._deep_copy(base)
            for i, item in enumerate(override):
                if i < len(result):
                    if isinstance(result[i], dict) and isinstance(item, dict):
                        result[i] = self._merge_two(result[i], item)
                    else:
                        result[i] = self._deep_copy(item)
                else:
                    result.append(self._deep_copy(item))
            return result
        
        return self._deep_copy(override)
    
    def _deep_copy(self, obj: Any) -> Any:
        """深拷贝对象"""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj
    
    def set_list_strategy(self, strategy: ListMergeStrategy) -> None:
        """设置列表合并策略"""
        self._list_strategy = strategy
    
    def set_preserve_none(self, preserve: bool) -> None:
        """设置是否保留 None 值"""
        self._preserve_none = preserve
