"""
数据库测试夹具
"""
import pytest
import tempfile
import os
from datetime import datetime, timedelta

from database.connection import init_database
from database.models.user import User
from database.models.node import Node, NodeStatus
from database.models.task import Task, TaskStatus
from database.models.experiment import Experiment, ExperimentStatus
from database.models.metric_record import MetricRecord


@pytest.fixture
def sqlite_test_config():
    """SQLite测试数据库配置"""
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    config = {
        'type': 'sqlite',
        'name': temp_db.name,
        'pool_size': 5,
        'echo': False
    }
    
    yield config
    
    # 清理
    if os.path.exists(temp_db.name):
        os.unlink(temp_db.name)


@pytest.fixture
def mysql_test_config():
    """MySQL测试数据库配置"""
    return {
        'type': 'mysql',
        'host': 'localhost',
        'port': 3306,
        'name': 'test_db',
        'user': 'test_user',
        'password': 'test_pass',
        'charset': 'utf8mb4',
        'pool_size': 5,
        'echo': False
    }


@pytest.fixture
def test_database(sqlite_test_config):
    """初始化的测试数据库"""
    db_conn = init_database(sqlite_test_config)
    db_conn.create_tables()
    yield db_conn
    db_conn.close()


@pytest.fixture
def sample_user_data():
    """示例用户数据"""
    return {
        'username': 'testuser',
        'email': 'test@example.com',
        'full_name': 'Test User',
        'password_hash': 'hashed_password_123',
        'is_active': True,
        'is_admin': False
    }


@pytest.fixture
def sample_admin_data():
    """示例管理员数据"""
    return {
        'username': 'admin',
        'email': 'admin@example.com',
        'full_name': 'Administrator',
        'password_hash': 'hashed_admin_password',
        'is_active': True,
        'is_admin': True
    }


@pytest.fixture
def sample_node_data():
    """示例节点数据"""
    return {
        'name': 'test_node',
        'host': 'localhost',
        'port': 8001,
        'status': NodeStatus.OFFLINE,
        'description': 'Test compute node',
        'cpu_cores': 4,
        'memory_gb': 8.0,
        'gpu_count': 1,
        'gpu_info': '{"type": "GTX 1080", "memory": "8GB"}',
        'max_concurrent_tasks': 2,
        'is_active': True
    }


@pytest.fixture
def sample_task_data():
    """示例任务数据"""
    return {
        'name': 'test_task',
        'description': 'Test machine learning task',
        'status': TaskStatus.PENDING,
        'priority': 3,
        'config': '{"model": "resnet50", "epochs": 100}',
        'command': 'python train.py --config config.yaml'
    }


@pytest.fixture
def sample_experiment_data():
    """示例实验数据"""
    return {
        'name': 'test_experiment',
        'description': 'Test experiment for image classification',
        'status': ExperimentStatus.CREATED,
        'config': '{"model": {"type": "resnet50", "pretrained": true}}',
        'hyperparameters': '{"lr": 0.001, "batch_size": 32}',
        'final_score': None,
        'best_epoch': None
    }


@pytest.fixture
def sample_metric_records():
    """示例指标记录数据"""
    base_time = datetime.utcnow()
    return [
        {
            'metric_name': 'train_loss',
            'metric_value': 2.5,
            'epoch': 1,
            'step': 100,
            'created_at': base_time
        },
        {
            'metric_name': 'val_loss',
            'metric_value': 2.3,
            'epoch': 1,
            'step': 100,
            'created_at': base_time + timedelta(minutes=1)
        },
        {
            'metric_name': 'train_accuracy',
            'metric_value': 0.1,
            'epoch': 1,
            'step': 100,
            'created_at': base_time
        },
        {
            'metric_name': 'val_accuracy',
            'metric_value': 0.12,
            'epoch': 1,
            'step': 100,
            'created_at': base_time + timedelta(minutes=1)
        },
        {
            'metric_name': 'train_loss',
            'metric_value': 1.8,
            'epoch': 2,
            'step': 200,
            'created_at': base_time + timedelta(hours=1)
        },
        {
            'metric_name': 'val_loss',
            'metric_value': 1.9,
            'epoch': 2,
            'step': 200,
            'created_at': base_time + timedelta(hours=1, minutes=1)
        },
        {
            'metric_name': 'train_accuracy',
            'metric_value': 0.35,
            'epoch': 2,
            'step': 200,
            'created_at': base_time + timedelta(hours=1)
        },
        {
            'metric_name': 'val_accuracy',
            'metric_value': 0.33,
            'epoch': 2,
            'step': 200,
            'created_at': base_time + timedelta(hours=1, minutes=1)
        }
    ]


@pytest.fixture
def complete_workflow_data(sample_user_data, sample_node_data, 
                          sample_task_data, sample_experiment_data):
    """完整工作流测试数据"""
    return {
        'user': sample_user_data,
        'node': sample_node_data,
        'task': sample_task_data,
        'experiment': sample_experiment_data
    }


@pytest.fixture
def mock_database_config():
    """模拟数据库配置（用于单元测试）"""
    return {
        'type': 'sqlite',
        'name': ':memory:',
        'pool_size': 1,
        'echo': False
    }


class DatabaseTestHelper:
    """数据库测试辅助类"""
    
    @staticmethod
    def create_test_user(session, user_data=None):
        """创建测试用户"""
        if user_data is None:
            user_data = {
                'username': 'testuser',
                'email': 'test@example.com',
                'password_hash': 'hashed_password'
            }
        
        user = User(**user_data)
        session.add(user)
        session.flush()
        session.refresh(user)
        return user
    
    @staticmethod
    def create_test_node(session, node_data=None):
        """创建测试节点"""
        if node_data is None:
            node_data = {
                'name': 'test_node',
                'host': 'localhost',
                'port': 8001,
                'status': NodeStatus.OFFLINE
            }
        
        node = Node(**node_data)
        session.add(node)
        session.flush()
        session.refresh(node)
        return node
    
    @staticmethod
    def create_test_task(session, task_data=None, user_id=None, node_id=None):
        """创建测试任务"""
        if task_data is None:
            task_data = {
                'name': 'test_task',
                'description': 'Test task',
                'status': TaskStatus.PENDING
            }
        
        if user_id:
            task_data['user_id'] = user_id
        if node_id:
            task_data['node_id'] = node_id
        
        task = Task(**task_data)
        session.add(task)
        session.flush()
        session.refresh(task)
        return task
    
    @staticmethod
    def create_test_experiment(session, exp_data=None, task_id=None, 
                             user_id=None, node_id=None):
        """创建测试实验"""
        if exp_data is None:
            exp_data = {
                'name': 'test_experiment',
                'description': 'Test experiment',
                'status': ExperimentStatus.CREATED
            }
        
        if task_id:
            exp_data['task_id'] = task_id
        if user_id:
            exp_data['user_id'] = user_id
        if node_id:
            exp_data['node_id'] = node_id
        
        experiment = Experiment(**exp_data)
        session.add(experiment)
        session.flush()
        session.refresh(experiment)
        return experiment
    
    @staticmethod
    def create_test_metrics(session, experiment_id, metric_records):
        """创建测试指标记录"""
        metrics = []
        for record_data in metric_records:
            record_data['experiment_id'] = experiment_id
            metric = MetricRecord(**record_data)
            session.add(metric)
            metrics.append(metric)
        
        session.flush()
        for metric in metrics:
            session.refresh(metric)
        
        return metrics


@pytest.fixture
def db_helper():
    """数据库测试辅助器"""
    return DatabaseTestHelper