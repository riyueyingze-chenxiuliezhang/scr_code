"""
# @Time: 2025/4/7 15:06
# @File: dqn_net.py
"""
from typing import List, Type

from torch import nn

from project.utils.impl.net import MLPNet, DuelingNet


class DqnNetFactory:
    """DQN网络工厂，根据配置动态创建指定类型的网络"""

    def __init__(self):
        # 注册支持的网络类型 (可扩展)
        self.network_registry = {
            "mlp": MLPNet,
            "dueling": DuelingNet
        }

    def build_network(
            self,
            input_size: int,
            output_size: int,
            hidden_layer: List[int] = None,
            net_name: str = "mlp"
    ) -> nn.Module:
        """
        根据配置构建网络
        Args:
            input_size: 输入维度
            output_size: 输出维度（动作数）
            hidden_layer: 隐藏层维度列表
            net_name: 使用的网络名称
        Returns:
            nn.Module: 实例化的网络模型
        """
        # 参数校验
        if net_name not in self.network_registry:
            raise KeyError(f"不支持的 net_name: {net_name}，可选: {list(self.network_registry.keys())}")

        # 获取网络类
        net_class = self.network_registry[net_name]

        # 构造参数 (可扩展其他网络的特殊参数)
        init_args = {
            "_input_size": input_size,
            "_output_size": output_size,
            "_hidden_layer": hidden_layer
        }

        # 创建网络实例
        return net_class(**init_args)

    def register_network(self, name: str, network_class: Type[nn.Module]):
        """注册自定义网络类型（扩展用）"""
        if not issubclass(network_class, nn.Module):
            raise TypeError("注册的网络类必须是 nn.Module 的子类")
        self.network_registry[name] = network_class


def dqn_net(input_size: int, output_size: int, hidden_layer: List[int] = None, net_name: str = "mlp"):
    factory = DqnNetFactory()
    return factory.build_network(input_size, output_size, hidden_layer, net_name)
