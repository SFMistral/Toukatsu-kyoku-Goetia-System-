"""
命令行覆盖参数解析器

解析点分隔路径覆盖语法，支持类型推断和特殊操作。
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class OverrideOperation(Enum):
    """覆盖操作类型"""
    SET = "set"           # 设置值
    DELETE = "delete"     # 删除键
    APPEND = "append"     # 追加到列表


@dataclass
class OverrideInstruction:
    """覆盖指令"""
    path: str
    operation: OverrideOperation
    value: Any = None
    explicit_type: str | None = None


class OverrideParser:
    """命令行覆盖参数解析器"""
    
    # 类型标注正则
    TYPE_PATTERN = re.compile(r"^(.+):(\w+)=(.+)$")
    # 删除操作正则
    DELETE_PATTERN = re.compile(r"^~(.+)$")
    # 追加操作正则
    APPEND_PATTERN = re.compile(r"^\+(.+)=(.+)$")
    # 普通设置正则
    SET_PATTERN = re.compile(r"^([^=]+)=(.+)$")

    def parse(self, overrides: list[str]) -> list[OverrideInstruction]:
        """
        解析覆盖参数列表
        
        Args:
            overrides: 覆盖参数字符串列表
        
        Returns:
            覆盖指令列表
        """
        instructions = []
        
        for override in overrides:
            override = override.strip()
            if not override:
                continue
            
            instruction = self._parse_single(override)
            instructions.append(instruction)
        
        return instructions
    
    def parse_to_dict(self, overrides: list[str]) -> dict:
        """
        解析覆盖参数并转换为嵌套字典
        
        Args:
            overrides: 覆盖参数字符串列表
        
        Returns:
            嵌套字典
        """
        instructions = self.parse(overrides)
        result = {}
        
        for inst in instructions:
            if inst.operation == OverrideOperation.DELETE:
                continue  # 删除操作需要特殊处理
            
            self._set_nested(result, inst.path, inst.value)
        
        return result
    
    def _parse_single(self, override: str) -> OverrideInstruction:
        """解析单个覆盖参数"""
        # 删除操作: ~key.path
        match = self.DELETE_PATTERN.match(override)
        if match:
            return OverrideInstruction(
                path=match.group(1),
                operation=OverrideOperation.DELETE,
            )
        
        # 追加操作: +key.path=[value]
        match = self.APPEND_PATTERN.match(override)
        if match:
            path = match.group(1)
            value = self._parse_value(match.group(2))
            return OverrideInstruction(
                path=path,
                operation=OverrideOperation.APPEND,
                value=value,
            )
        
        # 带类型标注: key.path:type=value
        match = self.TYPE_PATTERN.match(override)
        if match:
            path = match.group(1)
            explicit_type = match.group(2)
            value = self._parse_value(match.group(3), explicit_type)
            return OverrideInstruction(
                path=path,
                operation=OverrideOperation.SET,
                value=value,
                explicit_type=explicit_type,
            )
        
        # 普通设置: key.path=value
        match = self.SET_PATTERN.match(override)
        if match:
            path = match.group(1)
            value = self._parse_value(match.group(2))
            return OverrideInstruction(
                path=path,
                operation=OverrideOperation.SET,
                value=value,
            )
        
        raise ValueError(f"无法解析覆盖参数: {override}")

    def _parse_value(self, value_str: str, explicit_type: str | None = None) -> Any:
        """解析值字符串，支持类型推断"""
        value_str = value_str.strip()
        
        # 显式类型转换
        if explicit_type:
            return self._convert_type(value_str, explicit_type)
        
        # 自动类型推断
        # 布尔值
        if value_str.lower() in ("true", "yes", "on"):
            return True
        if value_str.lower() in ("false", "no", "off"):
            return False
        
        # None
        if value_str.lower() in ("null", "none", "~"):
            return None
        
        # 列表 [a, b, c]
        if value_str.startswith("[") and value_str.endswith("]"):
            return self._parse_list(value_str)
        
        # 字典 {a: 1, b: 2}
        if value_str.startswith("{") and value_str.endswith("}"):
            return self._parse_dict(value_str)
        
        # 数字
        try:
            if "." in value_str or "e" in value_str.lower():
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass
        
        # 字符串（去除引号）
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]
        
        return value_str
    
    def _convert_type(self, value_str: str, type_name: str) -> Any:
        """显式类型转换"""
        type_map = {
            "int": int,
            "float": float,
            "str": str,
            "bool": lambda x: x.lower() in ("true", "yes", "on", "1"),
        }
        
        if type_name in type_map:
            return type_map[type_name](value_str)
        
        raise ValueError(f"不支持的类型: {type_name}")
    
    def _parse_list(self, value_str: str) -> list:
        """解析列表字符串"""
        inner = value_str[1:-1].strip()
        if not inner:
            return []
        
        items = []
        for item in inner.split(","):
            items.append(self._parse_value(item.strip()))
        return items
    
    def _parse_dict(self, value_str: str) -> dict:
        """解析字典字符串"""
        inner = value_str[1:-1].strip()
        if not inner:
            return {}
        
        result = {}
        for pair in inner.split(","):
            if ":" in pair:
                key, val = pair.split(":", 1)
                result[key.strip()] = self._parse_value(val.strip())
        return result
    
    def _set_nested(self, d: dict, path: str, value: Any) -> None:
        """设置嵌套字典的值"""
        keys = path.split(".")
        current = d
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
