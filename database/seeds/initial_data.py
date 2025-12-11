"""
初始数据种子
"""
import logging
from database.connection import get_db_session
from database.models.user import User
from database.models.node import Node, NodeStatus

logger = logging.getLogger(__name__)

def create_initial_data():
    """创建初始数据"""
    try:
        with get_db_session() as session:
            # 创建默认管理员用户
            admin_user = session.query(User).filter(User.username == 'admin').first()
            if not admin_user:
                admin_user = User(
                    username='admin',
                    email='admin@example.com',
                    full_name='系统管理员',
                    is_active=True,
                    is_admin=True,
                    password_hash='$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3bp.Gm.F5.'  # admin123
                )
                session.add(admin_user)
                logger.info("创建默认管理员用户")
            
            # 创建默认普通用户
            normal_user = session.query(User).filter(User.username == 'user').first()
            if not normal_user:
                normal_user = User(
                    username='user',
                    email='user@example.com',
                    full_name='普通用户',
                    is_active=True,
                    is_admin=False,
                    password_hash='$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3bp.Gm.F5.'  # user123
                )
                session.add(normal_user)
                logger.info("创建默认普通用户")
            
            # 创建本地节点
            local_node = session.query(Node).filter(Node.name == 'local-node').first()
            if not local_node:
                local_node = Node(
                    name='local-node',
                    host='localhost',
                    port=8001,
                    status=NodeStatus.OFFLINE,
                    description='本地计算节点',
                    cpu_cores=4,
                    memory_gb=8.0,
                    gpu_count=0,
                    max_concurrent_tasks=2,
                    is_active=True
                )
                session.add(local_node)
                logger.info("创建本地计算节点")
            
            session.commit()
            logger.info("初始数据创建完成")
            
    except Exception as e:
        logger.error(f"创建初始数据失败: {e}")
        raise