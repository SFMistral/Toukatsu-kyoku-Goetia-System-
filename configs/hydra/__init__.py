"""
Hydra 核心功能模块

提供配置加载、组合、校验、覆盖解析和插值处理。
"""

from configs.hydra.config_loader import ConfigLoader
from configs.hydra.config_composer import ConfigComposer
from configs.hydra.config_validator import ConfigValidator
from configs.hydra.override_parser import OverrideParser
from configs.hydra.interpolation import InterpolationResolver

# 便捷函数
_loader = None
_composer = None
_validator = None


def _get_loader() -> ConfigLoader:
    global _loader
    if _loader is None:
        _loader = ConfigLoader()
    return _loader


def _get_composer() -> ConfigComposer:
    global _composer
    if _composer is None:
        _composer = ConfigComposer()
    return _composer


def _get_validator() -> ConfigValidator:
    global _validator
    if _validator is None:
        _validator = ConfigValidator()
    return _validator


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """
    加载配置文件，应用覆盖参数，返回校验后的配置对象
    
    Args:
        path: 配置文件路径
        overrides: 覆盖参数列表，如 ["optimizer.lr=0.001", "batch_size=32"]
    
    Returns:
        校验后的配置字典
    """
    loader = _get_loader()
    composer = _get_composer()
    validator = _get_validator()
    
    # 加载配置
    config = loader.load(path)
    
    # 应用覆盖
    if overrides:
        parser = OverrideParser()
        override_dict = parser.parse_to_dict(overrides)
        config = composer.merge(config, override_dict)
    
    # 解析插值
    resolver = InterpolationResolver()
    config = resolver.resolve(config)
    
    # 校验
    result = validator.validate(config)
    if not result.is_valid:
        raise ValueError(f"配置校验失败: {result.errors}")
    
    return config


def validate_config(config: dict, schema: str | None = None):
    """
    校验配置合法性
    
    Args:
        config: 配置字典
        schema: 指定的 schema 类型
    
    Returns:
        ValidationResult 对象
    """
    validator = _get_validator()
    return validator.validate(config, schema)


def merge_configs(base: dict, *others: dict) -> dict:
    """
    合并多个配置
    
    Args:
        base: 基础配置
        *others: 其他配置
    
    Returns:
        合并后的配置字典
    """
    composer = _get_composer()
    result = base.copy()
    for other in others:
        result = composer.merge(result, other)
    return result


def dump_config(config: dict, path: str, format: str = "yaml") -> None:
    """
    将配置导出到文件
    
    Args:
        config: 配置字典
        path: 输出文件路径
        format: 输出格式 (yaml/json)
    """
    import json
    import yaml
    
    with open(path, "w", encoding="utf-8") as f:
        if format == "yaml":
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif format == "json":
            json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的格式: {format}")


def resolve_interpolation(config: dict) -> dict:
    """
    解析配置中的所有插值变量
    
    Args:
        config: 配置字典
    
    Returns:
        解析后的配置字典
    """
    resolver = InterpolationResolver()
    return resolver.resolve(config)


__all__ = [
    "ConfigLoader",
    "ConfigComposer",
    "ConfigValidator",
    "OverrideParser",
    "InterpolationResolver",
    "load_config",
    "validate_config",
    "merge_configs",
    "dump_config",
    "resolve_interpolation",
]
