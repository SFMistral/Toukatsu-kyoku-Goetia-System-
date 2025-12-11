"""
数据库初始化脚本
"""
import logging
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.connection import init_database, get_db_connection
from database.seeds import create_initial_data, create_demo_data

logger = logging.getLogger(__name__)

def init_database_with_config(config):
    """使用配置初始化数据库"""
    try:
        # 初始化数据库连接
        db_conn = init_database(config)
        
        # 创建所有表
        db_conn.create_tables()
        logger.info("数据库表创建成功")
        
        # 创建初始数据
        create_initial_data()
        logger.info("初始数据创建成功")
        
        return True
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        return False

def create_demo_database(config):
    """创建包含演示数据的数据库"""
    try:
        # 先初始化基础数据库
        if not init_database_with_config(config):
            return False
        
        # 创建演示数据
        create_demo_data()
        logger.info("演示数据创建成功")
        
        return True
        
    except Exception as e:
        logger.error(f"演示数据库创建失败: {e}")
        return False

def reset_database(config):
    """重置数据库（删除所有表并重新创建）"""
    try:
        # 初始化数据库连接
        db_conn = init_database(config)
        
        # 删除所有表
        db_conn.drop_tables()
        logger.info("数据库表删除成功")
        
        # 重新创建表和数据
        return init_database_with_config(config)
        
    except Exception as e:
        logger.error(f"数据库重置失败: {e}")
        return False

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 示例配置
    config = {
        'type': 'mysql',
        'host': '127.0.0.1',
        'port': 19630,
        'name': 'test-dev',
        'user': 'zzxl',
        'password': 'user123456',
        'pool_size': 10,
        'echo': False,
        'charset': 'utf8mb4',
        'connect_args': {
            'autocommit': False,
            'check_same_thread': False
        }
    }
    
    # 根据命令行参数执行不同操作
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'init':
            success = init_database_with_config(config)
        elif command == 'demo':
            success = create_demo_database(config)
        elif command == 'reset':
            success = reset_database(config)
        else:
            print("用法: python init_db.py [init|demo|reset]")
            sys.exit(1)
        
        if success:
            print(f"数据库操作 '{command}' 完成")
        else:
            print(f"数据库操作 '{command}' 失败")
            sys.exit(1)
    else:
        print("用法: python init_db.py [init|demo|reset]")
        print("  init  - 初始化数据库和基础数据")
        print("  demo  - 创建包含演示数据的数据库")
        print("  reset - 重置数据库（删除所有数据）")