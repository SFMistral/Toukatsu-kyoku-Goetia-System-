# -*- coding: utf-8 -*-
"""
可视化钩子模块

提供训练过程可视化功能，支持预测结果、特征图、注意力图等可视化。
"""

import os
from typing import Dict, Any, Optional, List, Tuple, Union, TYPE_CHECKING

import torch
import numpy as np

from .base_hook import BaseHook, HookPriority
from registry import HOOKS

if TYPE_CHECKING:
    from typing import Any as RunnerType


class VisualizationHook(BaseHook):
    """
    通用可视化钩子
    
    可视化预测结果、特征图等。
    
    Args:
        interval: 可视化间隔（iter）
        draw_gt: 是否绘制真值
        draw_pred: 是否绘制预测
        show: 是否显示窗口
        save: 是否保存文件
        out_dir: 保存目录
        max_samples: 每次最大样本数
        score_thr: 置信度阈值（检测任务）
        backend: 可视化后端
        
    Example:
        >>> hook = VisualizationHook(
        ...     interval=500,
        ...     max_samples=8,
        ...     save=True
        ... )
    """
    
    priority = HookPriority.LOW
    
    def __init__(
        self,
        interval: int = 500,
        draw_gt: bool = True,
        draw_pred: bool = True,
        show: bool = False,
        save: bool = True,
        out_dir: Optional[str] = None,
        max_samples: int = 16,
        score_thr: float = 0.3,
        backend: str = 'matplotlib',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.interval = interval
        self.draw_gt = draw_gt
        self.draw_pred = draw_pred
        self.show = show
        self.save = save
        self.out_dir = out_dir
        self.max_samples = max_samples
        self.score_thr = score_thr
        self.backend = backend
        
    def before_run(self, runner: 'RunnerType') -> None:
        """初始化保存目录"""
        if self.save:
            if self.out_dir is None:
                self.out_dir = os.path.join(
                    getattr(runner, 'work_dir', '.'),
                    'visualizations'
                )
            os.makedirs(self.out_dir, exist_ok=True)
            
    def after_train_iter(self, runner: 'RunnerType') -> None:
        """训练迭代后可视化"""
        if not self.every_n_iters(runner, self.interval):
            return
            
        self._visualize(runner, mode='train')
        
    def after_val_iter(self, runner: 'RunnerType') -> None:
        """验证迭代后可视化"""
        if not self.every_n_iters(runner, self.interval):
            return
            
        self._visualize(runner, mode='val')
        
    def _visualize(self, runner: 'RunnerType', mode: str = 'train') -> None:
        """执行可视化"""
        if not hasattr(runner, 'data_batch') or not hasattr(runner, 'outputs'):
            return
            
        try:
            import matplotlib.pyplot as plt
            
            data_batch = runner.data_batch
            outputs = runner.outputs
            
            # 获取图像和标签
            images = data_batch.get('inputs', data_batch.get('img'))
            if images is None:
                return
                
            # 限制样本数
            num_samples = min(len(images), self.max_samples)
            
            # 创建图像网格
            fig, axes = self._create_grid(num_samples)
            
            for i in range(num_samples):
                ax = axes.flat[i] if num_samples > 1 else axes
                
                # 显示图像
                img = self._tensor_to_image(images[i])
                ax.imshow(img)
                ax.axis('off')
                
                # 添加标题
                title = f'{mode} iter {runner.iter + 1}'
                if 'loss' in outputs:
                    title += f' loss: {outputs["loss"]:.4f}'
                ax.set_title(title, fontsize=8)
                
            plt.tight_layout()
            
            # 保存或显示
            if self.save:
                save_path = os.path.join(
                    self.out_dir,
                    f'{mode}_iter_{runner.iter + 1}.png'
                )
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                
            if self.show:
                plt.show()
            else:
                plt.close()
                
        except ImportError:
            print("Warning: matplotlib not installed, visualization disabled")
        except Exception as e:
            print(f"Visualization error: {e}")
            
    def _create_grid(self, num_samples: int) -> Tuple:
        """创建图像网格"""
        import matplotlib.pyplot as plt
        
        cols = min(4, num_samples)
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        return fig, axes
        
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """将张量转换为图像"""
        if tensor.dim() == 4:
            tensor = tensor[0]
            
        # CHW -> HWC
        if tensor.dim() == 3 and tensor.shape[0] in [1, 3]:
            tensor = tensor.permute(1, 2, 0)
            
        # 转换为 numpy
        img = tensor.detach().cpu().numpy()
        
        # 归一化到 [0, 1]
        if img.max() > 1:
            img = img / 255.0
        img = np.clip(img, 0, 1)
        
        # 灰度图转 RGB
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.concatenate([img] * 3, axis=-1)
            
        return img


class DetVisualizationHook(VisualizationHook):
    """
    检测可视化钩子
    
    可视化目标检测结果。
    
    Args:
        class_names: 类别名称列表
        palette: 颜色调色板
        bbox_color: 边框颜色
        text_color: 文本颜色
        thickness: 线条粗细
        **kwargs: VisualizationHook 参数
        
    Example:
        >>> hook = DetVisualizationHook(
        ...     class_names=['cat', 'dog'],
        ...     interval=500
        ... )
    """
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        palette: Optional[List[Tuple[int, int, int]]] = None,
        bbox_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.class_names = class_names
        self.palette = palette or self._get_default_palette()
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.thickness = thickness
        
    def _visualize(self, runner: 'RunnerType', mode: str = 'train') -> None:
        """执行检测可视化"""
        if not hasattr(runner, 'data_batch') or not hasattr(runner, 'outputs'):
            return
            
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            data_batch = runner.data_batch
            outputs = runner.outputs
            
            images = data_batch.get('inputs', data_batch.get('img'))
            if images is None:
                return
                
            # 获取预测框
            pred_bboxes = outputs.get('bboxes', outputs.get('pred_bboxes'))
            pred_labels = outputs.get('labels', outputs.get('pred_labels'))
            pred_scores = outputs.get('scores', outputs.get('pred_scores'))
            
            # 获取真值框
            gt_bboxes = data_batch.get('gt_bboxes')
            gt_labels = data_batch.get('gt_labels')
            
            num_samples = min(len(images), self.max_samples)
            fig, axes = self._create_grid(num_samples)
            
            for i in range(num_samples):
                ax = axes.flat[i] if num_samples > 1 else axes
                
                img = self._tensor_to_image(images[i])
                ax.imshow(img)
                
                # 绘制真值框
                if self.draw_gt and gt_bboxes is not None:
                    self._draw_bboxes(ax, gt_bboxes[i], gt_labels[i] if gt_labels else None,
                                     color='green', linestyle='--')
                    
                # 绘制预测框
                if self.draw_pred and pred_bboxes is not None:
                    scores = pred_scores[i] if pred_scores is not None else None
                    self._draw_bboxes(ax, pred_bboxes[i], pred_labels[i] if pred_labels else None,
                                     scores=scores, color='red')
                    
                ax.axis('off')
                
            plt.tight_layout()
            
            if self.save:
                save_path = os.path.join(
                    self.out_dir,
                    f'det_{mode}_iter_{runner.iter + 1}.png'
                )
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                
            if self.show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Detection visualization error: {e}")
            
    def _draw_bboxes(
        self,
        ax,
        bboxes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
        color: str = 'red',
        linestyle: str = '-'
    ) -> None:
        """绘制边界框"""
        import matplotlib.patches as patches
        
        if bboxes is None or len(bboxes) == 0:
            return
            
        bboxes = bboxes.detach().cpu().numpy()
        if labels is not None:
            labels = labels.detach().cpu().numpy()
        if scores is not None:
            scores = scores.detach().cpu().numpy()
            
        for j, bbox in enumerate(bboxes):
            # 过滤低置信度
            if scores is not None and scores[j] < self.score_thr:
                continue
                
            x1, y1, x2, y2 = bbox[:4]
            width, height = x2 - x1, y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=self.thickness,
                edgecolor=color,
                facecolor='none',
                linestyle=linestyle
            )
            ax.add_patch(rect)
            
            # 添加标签
            label_text = ''
            if labels is not None and self.class_names:
                label_idx = int(labels[j])
                if label_idx < len(self.class_names):
                    label_text = self.class_names[label_idx]
            if scores is not None:
                label_text += f' {scores[j]:.2f}'
                
            if label_text:
                ax.text(x1, y1 - 2, label_text, fontsize=6,
                       color='white', backgroundcolor=color)
                       
    def _get_default_palette(self) -> List[Tuple[int, int, int]]:
        """获取默认调色板"""
        return [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128),
        ]


class SegVisualizationHook(VisualizationHook):
    """
    分割可视化钩子
    
    可视化语义分割结果。
    
    Args:
        class_names: 类别名称列表
        palette: 分割调色板
        opacity: 叠加透明度
        show_edge: 是否显示边缘
        **kwargs: VisualizationHook 参数
        
    Example:
        >>> hook = SegVisualizationHook(
        ...     class_names=['background', 'person', 'car'],
        ...     opacity=0.5
        ... )
    """
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        palette: Optional[List[Tuple[int, int, int]]] = None,
        opacity: float = 0.5,
        show_edge: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.class_names = class_names
        self.palette = palette or self._get_default_palette()
        self.opacity = opacity
        self.show_edge = show_edge
        
    def _visualize(self, runner: 'RunnerType', mode: str = 'train') -> None:
        """执行分割可视化"""
        if not hasattr(runner, 'data_batch') or not hasattr(runner, 'outputs'):
            return
            
        try:
            import matplotlib.pyplot as plt
            
            data_batch = runner.data_batch
            outputs = runner.outputs
            
            images = data_batch.get('inputs', data_batch.get('img'))
            if images is None:
                return
                
            # 获取预测掩码
            pred_masks = outputs.get('masks', outputs.get('pred_masks', outputs.get('seg_logits')))
            
            # 获取真值掩码
            gt_masks = data_batch.get('gt_masks', data_batch.get('gt_semantic_seg'))
            
            num_samples = min(len(images), self.max_samples)
            
            # 创建更大的网格（原图、预测、真值）
            cols = 3 if self.draw_gt and gt_masks is not None else 2
            fig, axes = plt.subplots(num_samples, cols, figsize=(cols * 4, num_samples * 4))
            
            if num_samples == 1:
                axes = axes.reshape(1, -1)
                
            for i in range(num_samples):
                img = self._tensor_to_image(images[i])
                
                # 原图
                axes[i, 0].imshow(img)
                axes[i, 0].set_title('Input', fontsize=10)
                axes[i, 0].axis('off')
                
                # 预测结果
                if pred_masks is not None:
                    pred_vis = self._visualize_mask(img, pred_masks[i])
                    axes[i, 1].imshow(pred_vis)
                    axes[i, 1].set_title('Prediction', fontsize=10)
                    axes[i, 1].axis('off')
                    
                # 真值
                if self.draw_gt and gt_masks is not None and cols > 2:
                    gt_vis = self._visualize_mask(img, gt_masks[i])
                    axes[i, 2].imshow(gt_vis)
                    axes[i, 2].set_title('Ground Truth', fontsize=10)
                    axes[i, 2].axis('off')
                    
            plt.tight_layout()
            
            if self.save:
                save_path = os.path.join(
                    self.out_dir,
                    f'seg_{mode}_iter_{runner.iter + 1}.png'
                )
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                
            if self.show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Segmentation visualization error: {e}")
            
    def _visualize_mask(
        self,
        image: np.ndarray,
        mask: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """可视化分割掩码"""
        # 转换掩码
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
            
        # 如果是 logits，取 argmax
        if mask.ndim == 3 and mask.shape[0] > 1:
            mask = mask.argmax(axis=0)
        elif mask.ndim == 3:
            mask = mask.squeeze(0)
            
        # 创建彩色掩码
        h, w = mask.shape[:2]
        color_mask = np.zeros((h, w, 3), dtype=np.float32)
        
        unique_labels = np.unique(mask)
        for label in unique_labels:
            if label < len(self.palette):
                color = np.array(self.palette[int(label)]) / 255.0
                color_mask[mask == label] = color
                
        # 调整图像大小以匹配掩码
        if image.shape[:2] != (h, w):
            from PIL import Image
            img_pil = Image.fromarray((image * 255).astype(np.uint8))
            img_pil = img_pil.resize((w, h))
            image = np.array(img_pil) / 255.0
            
        # 叠加
        result = image * (1 - self.opacity) + color_mask * self.opacity
        result = np.clip(result, 0, 1)
        
        # 显示边缘
        if self.show_edge:
            result = self._add_edge(result, mask)
            
        return result
        
    def _add_edge(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """添加分割边缘"""
        try:
            from scipy import ndimage
            
            # 计算边缘
            edges = np.zeros_like(mask, dtype=bool)
            for label in np.unique(mask):
                label_mask = mask == label
                eroded = ndimage.binary_erosion(label_mask)
                edges |= (label_mask & ~eroded)
                
            # 在边缘处绘制白线
            image[edges] = [1, 1, 1]
            
        except ImportError:
            pass
            
        return image
        
    def _get_default_palette(self) -> List[Tuple[int, int, int]]:
        """获取默认分割调色板"""
        return [
            (0, 0, 0),       # background
            (128, 0, 0),     # class 1
            (0, 128, 0),     # class 2
            (128, 128, 0),   # class 3
            (0, 0, 128),     # class 4
            (128, 0, 128),   # class 5
            (0, 128, 128),   # class 6
            (128, 128, 128), # class 7
            (64, 0, 0),      # class 8
            (192, 0, 0),     # class 9
            (64, 128, 0),    # class 10
            (192, 128, 0),   # class 11
            (64, 0, 128),    # class 12
            (192, 0, 128),   # class 13
            (64, 128, 128),  # class 14
            (192, 128, 128), # class 15
            (0, 64, 0),      # class 16
            (128, 64, 0),    # class 17
            (0, 192, 0),     # class 18
            (128, 192, 0),   # class 19
            (0, 64, 128),    # class 20
        ]


# 注册钩子（如果尚未注册）
def _safe_register(name, cls, **kwargs):
    if not HOOKS.contains(name):
        HOOKS.register(name, cls, **kwargs)
    else:
        HOOKS._components[name].cls = cls

_safe_register('VisualizationHook', VisualizationHook, 
               priority=HookPriority.LOW, category='visualization')
_safe_register('DetVisualizationHook', DetVisualizationHook, 
               priority=HookPriority.LOW, category='visualization')
_safe_register('SegVisualizationHook', SegVisualizationHook, 
               priority=HookPriority.LOW, category='visualization')
