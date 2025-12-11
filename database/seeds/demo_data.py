"""
演示数据种子
"""
import logging
import json
from datetime import datetime, timedelta
from database.connection import get_db_session
from database.models.user import User
from database.models.node import Node, NodeStatus
from database.models.task import Task, TaskStatus
from database.models.experiment import Experiment, ExperimentStatus
from database.models.metric_record import MetricRecord

logger = logging.getLogger(__name__)

def create_demo_data():
    """创建演示数据"""
    try:
        with get_db_session() as session:
            # 获取用户和节点
            admin_user = session.query(User).filter(User.username == 'admin').first()
            local_node = session.query(Node).filter(Node.name == 'local-node').first()
            
            if not admin_user or not local_node:
                logger.error("请先创建初始数据")
                return
            
            # 创建演示任务
            demo_task = Task(
                name='图像分类训练任务',
                description='使用ResNet50进行CIFAR-10图像分类',
                status=TaskStatus.COMPLETED,
                priority=3,
                config=json.dumps({
                    'model': 'resnet50',
                    'dataset': 'cifar10',
                    'epochs': 100,
                    'batch_size': 32,
                    'learning_rate': 0.001
                }),
                command='python train.py --config config.yaml',
                node_id=local_node.id,
                user_id=admin_user.id
            )
            session.add(demo_task)
            session.flush()
            
            # 创建演示实验
            demo_experiment = Experiment(
                name='ResNet50-CIFAR10-Exp1',
                description='ResNet50在CIFAR-10上的基准实验',
                status=ExperimentStatus.COMPLETED,
                config=json.dumps({
                    'model_config': {
                        'architecture': 'resnet50',
                        'num_classes': 10,
                        'pretrained': True
                    },
                    'training_config': {
                        'optimizer': 'adam',
                        'learning_rate': 0.001,
                        'weight_decay': 1e-4,
                        'epochs': 100
                    }
                }),
                hyperparameters=json.dumps({
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'dropout': 0.5,
                    'weight_decay': 1e-4
                }),
                final_score=0.9234,
                best_epoch=87,
                task_id=demo_task.id,
                node_id=local_node.id,
                user_id=admin_user.id
            )
            session.add(demo_experiment)
            session.flush()
            
            # 创建演示指标记录
            base_time = datetime.utcnow() - timedelta(hours=2)
            
            # 训练损失
            for epoch in range(1, 101):
                # 模拟训练损失下降
                train_loss = 2.5 * (0.95 ** epoch) + 0.1
                val_loss = 2.3 * (0.94 ** epoch) + 0.15
                
                # 模拟准确率提升
                train_acc = min(0.99, 0.1 + 0.89 * (1 - 0.95 ** epoch))
                val_acc = min(0.95, 0.08 + 0.87 * (1 - 0.94 ** epoch))
                
                # 添加一些随机波动
                import random
                train_loss += random.uniform(-0.05, 0.05)
                val_loss += random.uniform(-0.05, 0.05)
                train_acc += random.uniform(-0.02, 0.02)
                val_acc += random.uniform(-0.02, 0.02)
                
                metrics = [
                    MetricRecord(
                        metric_name='train_loss',
                        metric_value=max(0.01, train_loss),
                        epoch=epoch,
                        experiment_id=demo_experiment.id,
                        created_at=base_time + timedelta(minutes=epoch * 2)
                    ),
                    MetricRecord(
                        metric_name='val_loss',
                        metric_value=max(0.01, val_loss),
                        epoch=epoch,
                        experiment_id=demo_experiment.id,
                        created_at=base_time + timedelta(minutes=epoch * 2 + 1)
                    ),
                    MetricRecord(
                        metric_name='train_accuracy',
                        metric_value=max(0.0, min(1.0, train_acc)),
                        epoch=epoch,
                        experiment_id=demo_experiment.id,
                        created_at=base_time + timedelta(minutes=epoch * 2)
                    ),
                    MetricRecord(
                        metric_name='val_accuracy',
                        metric_value=max(0.0, min(1.0, val_acc)),
                        epoch=epoch,
                        experiment_id=demo_experiment.id,
                        created_at=base_time + timedelta(minutes=epoch * 2 + 1)
                    )
                ]
                
                session.add_all(metrics)
                
                # 每10个epoch提交一次
                if epoch % 10 == 0:
                    session.flush()
            
            session.commit()
            logger.info("演示数据创建完成")
            
    except Exception as e:
        logger.error(f"创建演示数据失败: {e}")
        raise