"""
配置插值处理器

支持变量引用、环境变量、内置变量等插值。
"""

import os
import re
from datetime import datetime
from typing import Any, Callable


class InterpolationResolver:
    """配置插值处理器"""
    
    # 插值模式: ${...}
    PATTERN = re.compile(r"\$\{([^}]+)\}")
    
    def __init__(self):
        self._resolvers: dict[str, Callable] = {}
        self._context: dict = {}
        self._resolving: set = set()  # 循环引用检测
        
        # 注册内置解析器
        self._register_builtin_resolvers()
    
    def _register_builtin_resolvers(self) -> None:
        """注册内置解析器"""
        self.register_resolver("env", self._resolve_env)
        self.register_resolver("now", self._resolve_now)
        self.register_resolver("file", self._resolve_file)
    
    def register_resolver(self, name: str, resolver: Callable) -> None:
        """注册自定义解析器"""
        self._resolvers[name] = resolver

    def resolve(self, config: dict, context: dict | None = None) -> dict:
        """
        解析配置中的所有插值
        
        Args:
            config: 配置字典
            context: 额外的上下文变量
        
        Returns:
            解析后的配置字典
        """
        self._context = config.copy()
        if context:
            self._context.update(context)
        
        self._resolving.clear()
        return self._resolve_value(config)
    
    def _resolve_value(self, value: Any) -> Any:
        """递归解析值"""
        if isinstance(value, dict):
            return {k: self._resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_value(item) for item in value]
        elif isinstance(value, str):
            return self._resolve_string(value)
        return value
    
    def _resolve_string(self, s: str) -> Any:
        """解析字符串中的插值"""
        # 完整匹配（返回原始类型）
        match = self.PATTERN.fullmatch(s)
        if match:
            return self._resolve_expression(match.group(1))
        
        # 部分匹配（字符串拼接）
        def replacer(m):
            result = self._resolve_expression(m.group(1))
            return str(result) if result is not None else ""
        
        return self.PATTERN.sub(replacer, s)
    
    def _resolve_expression(self, expr: str) -> Any:
        """解析单个插值表达式"""
        expr = expr.strip()
        
        # 循环引用检测
        if expr in self._resolving:
            raise ValueError(f"检测到循环引用: {expr}")
        
        self._resolving.add(expr)
        
        try:
            # 带解析器前缀: env:VAR, now:%Y%m%d
            if ":" in expr:
                prefix, arg = expr.split(":", 1)
                if prefix in self._resolvers:
                    return self._resolvers[prefix](arg)
            
            # 变量引用: model.backbone.type
            return self._resolve_path(expr)
        finally:
            self._resolving.discard(expr)
    
    def _resolve_path(self, path: str) -> Any:
        """解析路径引用"""
        keys = path.split(".")
        current = self._context
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        # 递归解析结果
        if isinstance(current, str) and self.PATTERN.search(current):
            return self._resolve_string(current)
        
        return current
    
    def _resolve_env(self, arg: str) -> str | None:
        """解析环境变量"""
        if "," in arg:
            var_name, default = arg.split(",", 1)
            return os.environ.get(var_name.strip(), default.strip())
        return os.environ.get(arg.strip())
    
    def _resolve_now(self, format_str: str) -> str:
        """解析当前时间"""
        return datetime.now().strftime(format_str)
    
    def _resolve_file(self, path: str) -> str:
        """读取文件内容"""
        try:
            with open(path.strip(), "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""
