"""
节点仓库
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from database.models.node import Node, NodeStatus
from database.connection import get_db_session

class NodeRepository:
    """节点数据访问层"""
    
    def create(self, node_data: Dict[str, Any]) -> Node:
        """创建节点"""
        with get_db_session() as session:
            node = Node(**node_data)
            session.add(node)
            session.flush()
            session.refresh(node)
            session.expunge(node)
            return node
    
    def get_by_id(self, node_id: int) -> Optional[Node]:
        """根据ID获取节点"""
        with get_db_session() as session:
            return session.query(Node).filter(Node.id == node_id).first()
    
    def get_by_name(self, name: str) -> Optional[Node]:
        """根据名称获取节点"""
        with get_db_session() as session:
            return session.query(Node).filter(Node.name == name).first()
    
    def get_by_host_port(self, host: str, port: int) -> Optional[Node]:
        """根据主机和端口获取节点"""
        with get_db_session() as session:
            return session.query(Node).filter(
                and_(Node.host == host, Node.port == port)
            ).first()
    
    def get_all(self, limit: int = 100, offset: int = 0) -> List[Node]:
        """获取所有节点"""
        with get_db_session() as session:
            return session.query(Node).offset(offset).limit(limit).all()
    
    def get_active_nodes(self) -> List[Node]:
        """获取激活的节点"""
        with get_db_session() as session:
            return session.query(Node).filter(Node.is_active == True).all()
    
    def get_online_nodes(self) -> List[Node]:
        """获取在线节点"""
        with get_db_session() as session:
            return session.query(Node).filter(
                and_(
                    Node.status == NodeStatus.ONLINE,
                    Node.is_active == True
                )
            ).all()
    
    def get_available_nodes(self) -> List[Node]:
        """获取可用节点（在线且非忙碌）"""
        with get_db_session() as session:
            return session.query(Node).filter(
                and_(
                    Node.status.in_([NodeStatus.ONLINE]),
                    Node.is_active == True
                )
            ).all()
    
    def get_by_status(self, status: NodeStatus) -> List[Node]:
        """根据状态获取节点"""
        with get_db_session() as session:
            return session.query(Node).filter(Node.status == status).all()
    
    def update(self, node_id: int, update_data: Dict[str, Any]) -> Optional[Node]:
        """更新节点"""
        with get_db_session() as session:
            node = session.query(Node).filter(Node.id == node_id).first()
            if node:
                for key, value in update_data.items():
                    setattr(node, key, value)
                session.flush()
                session.refresh(node)
                session.expunge(node)
            return node
    
    def update_status(self, node_id: int, status: NodeStatus) -> Optional[Node]:
        """更新节点状态"""
        return self.update(node_id, {'status': status})
    
    def update_resource_usage(self, node_id: int, cpu_usage: float, 
                            memory_usage: float, gpu_usage: float = 0.0) -> Optional[Node]:
        """更新资源使用情况"""
        return self.update(node_id, {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'gpu_usage': gpu_usage
        })
    
    def delete(self, node_id: int) -> bool:
        """删除节点"""
        with get_db_session() as session:
            node = session.query(Node).filter(Node.id == node_id).first()
            if node:
                session.delete(node)
                return True
            return False
    
    def search(self, keyword: str, limit: int = 50) -> List[Node]:
        """搜索节点"""
        with get_db_session() as session:
            return session.query(Node).filter(
                or_(
                    Node.name.contains(keyword),
                    Node.description.contains(keyword),
                    Node.host.contains(keyword)
                )
            ).limit(limit).all()
    
    def count_by_status(self) -> Dict[str, int]:
        """按状态统计节点数量"""
        with get_db_session() as session:
            result = {}
            for status in NodeStatus:
                count = session.query(Node).filter(Node.status == status).count()
                result[status.value] = count
            return result