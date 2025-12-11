"""
模拟节点夹具
"""
import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any


@pytest.fixture
def mock_node_info():
    """模拟节点信息"""
    return {
        'name': 'mock_node_1',
        'host': '192.168.1.100',
        'port': 8001,
        'status': 'online',
        'cpu_cores': 8,
        'memory_gb': 32.0,
        'gpu_count': 2,
        'gpu_info': [
            {'name': 'NVIDIA RTX 3080', 'memory_gb': 10},
            {'name': 'NVIDIA RTX 3080', 'memory_gb': 10}
        ],
        'max_concurrent_tasks': 4
    }


@pytest.fixture
def mock_node_list():
    """模拟节点列表"""
    return [
        {
            'name': 'node_1',
            'host': '192.168.1.101',
            'port': 8001,
            'status': 'online',
            'cpu_cores': 8,
            'memory_gb': 32.0,
            'gpu_count': 2
        },
        {
            'name': 'node_2',
            'host': '192.168.1.102',
            'port': 8001,
            'status': 'online',
            'cpu_cores': 16,
            'memory_gb': 64.0,
            'gpu_count': 4
        },
        {
            'name': 'node_3',
            'host': '192.168.1.103',
            'port': 8001,
            'status': 'offline',
            'cpu_cores': 4,
            'memory_gb': 16.0,
            'gpu_count': 1
        }
    ]


@pytest.fixture
def mock_node_manager():
    """模拟节点管理器"""
    manager = Mock()
    manager.get_online_nodes.return_value = []
    manager.get_available_nodes.return_value = []
    manager.register_node.return_value = True
    manager.unregister_node.return_value = True
    manager.update_node_status.return_value = True
    return manager


@pytest.fixture
def mock_node_connection():
    """模拟节点连接"""
    connection = Mock()
    connection.is_connected = True
    connection.send_task.return_value = {'status': 'accepted', 'task_id': 1}
    connection.get_status.return_value = {'status': 'online', 'load': 0.5}
    connection.heartbeat.return_value = True
    return connection


@pytest.fixture
def mock_resource_usage():
    """模拟资源使用情况"""
    return {
        'cpu_usage': 45.5,
        'memory_usage': 60.2,
        'gpu_usage': [
            {'index': 0, 'usage': 80.0, 'memory_used': 8.5},
            {'index': 1, 'usage': 75.0, 'memory_used': 7.8}
        ],
        'disk_usage': 55.0,
        'network_io': {
            'bytes_sent': 1024000,
            'bytes_recv': 2048000
        }
    }