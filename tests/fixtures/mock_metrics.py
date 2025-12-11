"""
模拟指标夹具
"""
import pytest
from typing import Dict, List, Any
from datetime import datetime, timedelta


@pytest.fixture
def mock_training_metrics():
    """模拟训练指标"""
    return {
        'epoch': 10,
        'step': 1000,
        'train_loss': 0.5,
        'train_accuracy': 0.85,
        'val_loss': 0.6,
        'val_accuracy': 0.82,
        'learning_rate': 0.001
    }


@pytest.fixture
def mock_metric_history():
    """模拟指标历史"""
    history = []
    base_time = datetime.utcnow() - timedelta(hours=2)
    
    for epoch in range(1, 51):
        # 模拟训练过程中的指标变化
        train_loss = 2.5 * (0.95 ** epoch) + 0.1
        val_loss = 2.3 * (0.94 ** epoch) + 0.15
        train_acc = min(0.99, 0.1 + 0.89 * (1 - 0.95 ** epoch))
        val_acc = min(0.95, 0.08 + 0.87 * (1 - 0.94 ** epoch))
        
        history.append({
            'epoch': epoch,
            'timestamp': (base_time + timedelta(minutes=epoch * 2)).isoformat(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'learning_rate': 0.001 * (0.95 ** (epoch // 10))
        })
    
    return history


@pytest.fixture
def mock_evaluation_metrics():
    """模拟评估指标"""
    return {
        'accuracy': 0.92,
        'precision': 0.91,
        'recall': 0.93,
        'f1_score': 0.92,
        'confusion_matrix': [
            [95, 5],
            [7, 93]
        ],
        'classification_report': {
            'class_0': {'precision': 0.93, 'recall': 0.95, 'f1-score': 0.94},
            'class_1': {'precision': 0.95, 'recall': 0.93, 'f1-score': 0.94}
        }
    }


@pytest.fixture
def mock_detection_metrics():
    """模拟检测指标"""
    return {
        'mAP': 0.75,
        'mAP_50': 0.85,
        'mAP_75': 0.70,
        'mAP_small': 0.55,
        'mAP_medium': 0.75,
        'mAP_large': 0.85,
        'per_class_ap': {
            'person': 0.82,
            'car': 0.78,
            'dog': 0.71
        }
    }


@pytest.fixture
def mock_segmentation_metrics():
    """模拟分割指标"""
    return {
        'mIoU': 0.72,
        'pixel_accuracy': 0.95,
        'mean_accuracy': 0.85,
        'per_class_iou': {
            'background': 0.95,
            'person': 0.78,
            'car': 0.72,
            'road': 0.85
        }
    }


@pytest.fixture
def mock_resource_metrics():
    """模拟资源指标"""
    return {
        'gpu_memory_used': 8.5,
        'gpu_memory_total': 10.0,
        'gpu_utilization': 85.0,
        'cpu_utilization': 45.0,
        'memory_used': 12.5,
        'memory_total': 32.0,
        'throughput': 150.0,  # samples/sec
        'batch_time': 0.025   # seconds
    }


@pytest.fixture
def mock_experiment_summary():
    """模拟实验摘要"""
    return {
        'experiment_id': 1,
        'experiment_name': 'resnet50_cifar10_exp1',
        'status': 'completed',
        'start_time': '2023-01-01T10:00:00',
        'end_time': '2023-01-01T12:30:00',
        'duration_hours': 2.5,
        'best_epoch': 45,
        'best_metrics': {
            'val_accuracy': 0.923,
            'val_loss': 0.245
        },
        'final_metrics': {
            'val_accuracy': 0.918,
            'val_loss': 0.258
        },
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'optimizer': 'Adam'
        }
    }