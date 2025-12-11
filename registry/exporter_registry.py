# -*- coding: utf-8 -*-
"""
模型导出器注册器模块

管理模型导出相关组件，支持多种导出格式。
"""

from typing import Dict, Any, Optional, List, Type, Tuple
from abc import ABC, abstractmethod
from .registry import Registry, ComponentSource


class ExporterRegistry(Registry):
    """模型导出器注册器"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._format_capabilities: Dict[str, Dict[str, Any]] = {}
        
    def register(
        self,
        name: Optional[str] = None,
        cls: Optional[Type] = None,
        supported_formats: Optional[List[str]] = None,
        requires: Optional[List[str]] = None,
        **kwargs
    ):
        """
        注册导出器组件
        
        Args:
            name: 组件名称
            cls: 组件类
            supported_formats: 支持的导出格式
            requires: 依赖的包
            **kwargs: 其他注册参数
        """
        result = super().register(name=name, cls=cls, **kwargs)
        
        actual_name = name or (cls.__name__ if cls else None)
        if actual_name:
            self._format_capabilities[actual_name] = {
                'formats': supported_formats or [],
                'requires': requires or []
            }
            
        return result
        
    def get_capabilities(self, name: str) -> Dict[str, Any]:
        """获取导出器能力信息"""
        return self._format_capabilities.get(name, {})
        
    def list_by_format(self, format_name: str) -> List[str]:
        """按支持的格式筛选导出器"""
        return [
            name for name, caps in self._format_capabilities.items()
            if format_name in caps.get('formats', [])
        ]
        
    def check_requirements(self, name: str) -> Tuple[bool, List[str]]:
        """
        检查导出器依赖是否满足
        
        Returns:
            (是否满足, 缺失的包列表)
        """
        caps = self._format_capabilities.get(name, {})
        requires = caps.get('requires', [])
        
        missing = []
        for pkg in requires:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
                
        return len(missing) == 0, missing


# 创建导出器注册器单例
EXPORTERS = ExporterRegistry('exporters', base_class=None)


class BaseExporter(ABC):
    """导出器基类"""
    
    def __init__(self, output_dir: str = 'exports'):
        self.output_dir = output_dir
        
    @abstractmethod
    def export(self, model, input_shape: Tuple[int, ...], output_path: str, **kwargs) -> str:
        """
        导出模型
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状
            output_path: 输出路径
            **kwargs: 额外参数
            
        Returns:
            导出文件路径
        """
        pass
        
    def validate(self, model, exported_path: str, input_shape: Tuple[int, ...]) -> bool:
        """
        验证导出的模型
        
        Args:
            model: 原始PyTorch模型
            exported_path: 导出文件路径
            input_shape: 输入形状
            
        Returns:
            验证是否通过
        """
        return True
        
    def _prepare_output_dir(self):
        """准备输出目录"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)


class ONNXExporter(BaseExporter):
    """ONNX格式导出器"""
    
    def __init__(
        self,
        opset_version: int = 11,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        do_constant_folding: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.opset_version = opset_version
        self.dynamic_axes = dynamic_axes
        self.do_constant_folding = do_constant_folding
        
    def export(
        self,
        model,
        input_shape: Tuple[int, ...],
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        import torch
        import os
        
        self._prepare_output_dir()
        
        if not output_path.endswith('.onnx'):
            output_path = output_path + '.onnx'
            
        full_path = os.path.join(self.output_dir, output_path)
        
        # 创建dummy输入
        dummy_input = torch.randn(1, *input_shape)
        
        # 设置模型为评估模式
        model.eval()
        
        # 导出
        torch.onnx.export(
            model,
            dummy_input,
            full_path,
            opset_version=self.opset_version,
            do_constant_folding=self.do_constant_folding,
            input_names=input_names or ['input'],
            output_names=output_names or ['output'],
            dynamic_axes=self.dynamic_axes,
            **kwargs
        )
        
        print(f"Model exported to {full_path}")
        return full_path
        
    def validate(self, model, exported_path: str, input_shape: Tuple[int, ...]) -> bool:
        try:
            import onnx
            import onnxruntime as ort
            import torch
            import numpy as np
            
            # 检查ONNX模型
            onnx_model = onnx.load(exported_path)
            onnx.checker.check_model(onnx_model)
            
            # 比较输出
            dummy_input = torch.randn(1, *input_shape)
            
            model.eval()
            with torch.no_grad():
                torch_output = model(dummy_input).numpy()
                
            ort_session = ort.InferenceSession(exported_path)
            ort_output = ort_session.run(
                None, 
                {'input': dummy_input.numpy()}
            )[0]
            
            # 检查输出是否接近
            return np.allclose(torch_output, ort_output, rtol=1e-3, atol=1e-5)
            
        except Exception as e:
            print(f"Validation failed: {e}")
            return False


class TorchScriptExporter(BaseExporter):
    """TorchScript格式导出器"""
    
    def __init__(
        self,
        method: str = 'trace',  # 'trace' or 'script'
        optimize: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.method = method
        self.optimize = optimize
        
    def export(
        self,
        model,
        input_shape: Tuple[int, ...],
        output_path: str,
        **kwargs
    ) -> str:
        import torch
        import os
        
        self._prepare_output_dir()
        
        if not output_path.endswith('.pt'):
            output_path = output_path + '.pt'
            
        full_path = os.path.join(self.output_dir, output_path)
        
        model.eval()
        
        if self.method == 'trace':
            dummy_input = torch.randn(1, *input_shape)
            scripted_model = torch.jit.trace(model, dummy_input)
        else:
            scripted_model = torch.jit.script(model)
            
        if self.optimize:
            scripted_model = torch.jit.optimize_for_inference(scripted_model)
            
        scripted_model.save(full_path)
        
        print(f"Model exported to {full_path}")
        return full_path
        
    def validate(self, model, exported_path: str, input_shape: Tuple[int, ...]) -> bool:
        try:
            import torch
            import numpy as np
            
            dummy_input = torch.randn(1, *input_shape)
            
            model.eval()
            with torch.no_grad():
                torch_output = model(dummy_input).numpy()
                
            loaded_model = torch.jit.load(exported_path)
            loaded_model.eval()
            with torch.no_grad():
                script_output = loaded_model(dummy_input).numpy()
                
            return np.allclose(torch_output, script_output, rtol=1e-5, atol=1e-7)
            
        except Exception as e:
            print(f"Validation failed: {e}")
            return False


class OpenVINOExporter(BaseExporter):
    """OpenVINO格式导出器（预留）"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def export(
        self,
        model,
        input_shape: Tuple[int, ...],
        output_path: str,
        **kwargs
    ) -> str:
        # 先导出为ONNX，再转换为OpenVINO
        onnx_exporter = ONNXExporter(output_dir=self.output_dir)
        onnx_path = onnx_exporter.export(model, input_shape, output_path + '_temp')
        
        try:
            from openvino.tools import mo
            
            if not output_path.endswith('.xml'):
                output_path = output_path + '.xml'
                
            import os
            full_path = os.path.join(self.output_dir, output_path)
            
            mo.convert_model(
                onnx_path,
                output_model=full_path
            )
            
            # 清理临时ONNX文件
            os.remove(onnx_path)
            
            print(f"Model exported to {full_path}")
            return full_path
            
        except ImportError:
            raise ImportError("OpenVINO is not installed. Please install openvino-dev package.")


class TensorRTExporter(BaseExporter):
    """TensorRT格式导出器（预留）"""
    
    def __init__(
        self,
        fp16: bool = False,
        int8: bool = False,
        max_workspace_size: int = 1 << 30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fp16 = fp16
        self.int8 = int8
        self.max_workspace_size = max_workspace_size
        
    def export(
        self,
        model,
        input_shape: Tuple[int, ...],
        output_path: str,
        **kwargs
    ) -> str:
        # 先导出为ONNX
        onnx_exporter = ONNXExporter(output_dir=self.output_dir)
        onnx_path = onnx_exporter.export(model, input_shape, output_path + '_temp')
        
        try:
            import tensorrt as trt
            
            if not output_path.endswith('.engine'):
                output_path = output_path + '.engine'
                
            import os
            full_path = os.path.join(self.output_dir, output_path)
            
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            with open(onnx_path, 'rb') as f:
                parser.parse(f.read())
                
            config = builder.create_builder_config()
            config.max_workspace_size = self.max_workspace_size
            
            if self.fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if self.int8:
                config.set_flag(trt.BuilderFlag.INT8)
                
            engine = builder.build_engine(network, config)
            
            with open(full_path, 'wb') as f:
                f.write(engine.serialize())
                
            # 清理临时ONNX文件
            os.remove(onnx_path)
            
            print(f"Model exported to {full_path}")
            return full_path
            
        except ImportError:
            raise ImportError("TensorRT is not installed.")


# 注册内置导出器
EXPORTERS.register(
    'ONNXExporter', ONNXExporter,
    supported_formats=['onnx'],
    requires=['torch'],
    category='standard',
    source=ComponentSource.BUILTIN
)

EXPORTERS.register(
    'TorchScriptExporter', TorchScriptExporter,
    supported_formats=['torchscript', 'pt'],
    requires=['torch'],
    category='standard',
    source=ComponentSource.BUILTIN
)

EXPORTERS.register(
    'OpenVINOExporter', OpenVINOExporter,
    supported_formats=['openvino', 'xml', 'bin'],
    requires=['openvino'],
    category='inference',
    source=ComponentSource.BUILTIN
)

EXPORTERS.register(
    'TensorRTExporter', TensorRTExporter,
    supported_formats=['tensorrt', 'engine'],
    requires=['tensorrt'],
    category='inference',
    source=ComponentSource.BUILTIN
)
