"""
# @Time: 2025/4/9 14:45
# @File: replay_memory.py
"""
from .base import BaseMemory
from .pro_pri import ProPriReplayBuffer


class ReplayMemoryFactory:
    """经验回放工厂"""

    def __init__(self):
        # 注册支持的网络类型 (可扩展)
        self.memory_registry = {
            "base": BaseMemory,
            "pri": ProPriReplayBuffer
        }

    def build_memory(
            self,
            capacity,
            memory_name: str = "base"
    ):
        """
        根据配置构建网络
        Args:
            capacity: 经验池的容量
            memory_name: 使用的经验池名称
        Returns:
            nn.Module: 实例化的经验池
        """
        # 参数校验
        if memory_name not in self.memory_registry:
            raise KeyError(f"不支持的 memory_name: {memory_name}，可选: {list(self.memory_registry.keys())}")

        # 获取经验池类
        memory_class = self.memory_registry[memory_name]

        # 构造参数 (可扩展其他网络的特殊参数)
        init_args = {
            "capacity": capacity,
        }

        # 创建网络实例
        return memory_class(**init_args)


def replay_memory(capacity, memory_name: str = "base"):
    factory = ReplayMemoryFactory()
    return factory.build_memory(capacity, memory_name)
