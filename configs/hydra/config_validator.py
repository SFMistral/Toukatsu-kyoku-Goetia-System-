"""
配置校验器

基于 Schema 进行类型校验、必填字段检查、取值范围校验等。
"""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ValidationError:
    """校验错误"""
    path: str
    message: str
    value: Any = None


@dataclass
class ValidationResult:
    """校验结果"""
    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    
    def add_error(self, path: str, message: str, value: Any = None) -> None:
        self.errors.append(ValidationError(path, message, value))
        self.is_valid = False
    
    def add_warning(self, path: str, message: str, value: Any = None) -> None:
        self.warnings.append(ValidationError(path, message, value))
    
    def merge(self, other: "ValidationResult") -> None:
        """合并另一个校验结果"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False


class ConfigValidator:
    """配置校验器"""
    
    def __init__(self):
        self._schemas: dict[str, Callable] = {}
        self._custom_validators: list[Callable] = []
    
    def register_schema(self, name: str, schema_func: Callable) -> None:
        """注册 Schema"""
        self._schemas[name] = schema_func
    
    def register_validator(self, validator: Callable) -> None:
        """注册自定义校验器"""
        self._custom_validators.append(validator)
    
    def validate(self, config: dict, schema: str | None = None) -> ValidationResult:
        """
        校验配置
        
        Args:
            config: 配置字典
            schema: 指定的 schema 名称
        
        Returns:
            ValidationResult 对象
        """
        result = ValidationResult(is_valid=True)
        
        # 基础类型校验
        self._validate_types(config, "", result)
        
        # Schema 校验
        if schema and schema in self._schemas:
            schema_result = self._schemas[schema](config)
            result.merge(schema_result)
        
        # 自动检测并校验各部分
        self._validate_sections(config, result)
        
        # 自定义校验器
        for validator in self._custom_validators:
            validator_result = validator(config)
            if validator_result:
                result.merge(validator_result)
        
        return result
    
    def _validate_types(self, obj: Any, path: str, result: ValidationResult) -> None:
        """递归校验类型"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                self._validate_types(value, new_path, result)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                self._validate_types(item, new_path, result)
    
    def _validate_sections(self, config: dict, result: ValidationResult) -> None:
        """校验配置各部分"""
        # 模型配置校验
        if "model" in config:
            self._validate_model(config["model"], "model", result)
        
        # 数据集配置校验
        if "dataset" in config:
            self._validate_dataset(config["dataset"], "dataset", result)
        
        # 训练配置校验
        if "training" in config:
            self._validate_training(config["training"], "training", result)
        
        # 优化器配置校验
        if "optimizer" in config:
            self._validate_optimizer(config["optimizer"], "optimizer", result)
        
        # 导出配置校验
        if "export" in config:
            self._validate_export(config["export"], "export", result)
    
    def _validate_model(self, model: dict, path: str, result: ValidationResult) -> None:
        """校验模型配置"""
        if not isinstance(model, dict):
            result.add_error(path, "模型配置必须是字典类型", model)
            return
        
        # type 字段
        if "type" not in model:
            result.add_warning(path, "建议指定模型类型 (type)")
        
        # backbone 校验
        if "backbone" in model:
            backbone = model["backbone"]
            if isinstance(backbone, dict):
                if "type" not in backbone:
                    result.add_error(f"{path}.backbone", "backbone 必须指定 type")
    
    def _validate_dataset(self, dataset: dict, path: str, result: ValidationResult) -> None:
        """校验数据集配置"""
        if not isinstance(dataset, dict):
            result.add_error(path, "数据集配置必须是字典类型", dataset)
            return
        
        # 必须有数据路径
        if "data_root" not in dataset and "data_roots" not in dataset:
            result.add_warning(path, "建议指定数据路径 (data_root 或 data_roots)")
    
    def _validate_training(self, training: dict, path: str, result: ValidationResult) -> None:
        """校验训练配置"""
        if not isinstance(training, dict):
            result.add_error(path, "训练配置必须是字典类型", training)
            return
        
        # epochs 范围
        if "epochs" in training:
            epochs = training["epochs"]
            if not isinstance(epochs, int) or epochs < 1 or epochs > 10000:
                result.add_error(f"{path}.epochs", "epochs 必须是 1-10000 的整数", epochs)
        
        # batch_size 范围
        if "batch_size" in training:
            bs = training["batch_size"]
            if not isinstance(bs, int) or bs < 1 or bs > 1024:
                result.add_error(f"{path}.batch_size", "batch_size 必须是 1-1024 的整数", bs)
        
        # 分布式配置依赖检查
        if "distributed" in training:
            dist = training["distributed"]
            if isinstance(dist, dict) and dist.get("enabled"):
                if "backend" not in dist:
                    result.add_error(
                        f"{path}.distributed",
                        "启用分布式训练时必须指定 backend"
                    )
    
    def _validate_optimizer(self, optimizer: dict, path: str, result: ValidationResult) -> None:
        """校验优化器配置"""
        if not isinstance(optimizer, dict):
            result.add_error(path, "优化器配置必须是字典类型", optimizer)
            return
        
        if "type" not in optimizer:
            result.add_warning(path, "建议指定优化器类型 (type)")
        
        # 学习率范围
        if "lr" in optimizer:
            lr = optimizer["lr"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                result.add_error(f"{path}.lr", "学习率必须是正数", lr)
    
    def _validate_export(self, export: dict, path: str, result: ValidationResult) -> None:
        """校验导出配置"""
        if not isinstance(export, dict):
            result.add_error(path, "导出配置必须是字典类型", export)
            return
        
        valid_formats = {"onnx", "tensorrt", "openvino", "coreml", "ncnn"}
        
        if "formats" in export:
            formats = export["formats"]
            if isinstance(formats, list):
                for i, fmt in enumerate(formats):
                    if isinstance(fmt, dict) and "type" in fmt:
                        if fmt["type"] not in valid_formats:
                            result.add_error(
                                f"{path}.formats[{i}].type",
                                f"不支持的导出格式: {fmt['type']}",
                                fmt["type"]
                            )
