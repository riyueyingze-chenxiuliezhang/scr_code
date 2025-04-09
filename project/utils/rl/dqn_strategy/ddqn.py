"""
# @Time: 2025/4/8 18:40
# @File: ddqn.py
"""
import numpy as np
import torch
import torch.nn.functional as F

from .strategy import BaseStrategy
from ..dqn_impl import DQNImpl


@DQNImpl.register_strategy("dd")
class DDqn(BaseStrategy):
    def __init__(self, context):
        self._context = context

    def sample(self):
        pass

    def predict(self, states, actions, rewards, next_states, dones):
        q_value = self._context.q_net(states).gather(1, actions)
        with torch.no_grad():
            """
            1. 用在线网络选择动作
            2. 用目标网络评估 Q 值
            """
            next_actions = self._context.q_net(next_states).max(1)[1].view(-1, 1)
            next_q_value = self._context.target_q_net(next_states).gather(1, next_actions).detach()
            target_q_value = next_q_value * self._context.gamma * (1 - dones) + rewards
        return q_value, target_q_value

    def update(self):
        pass

    def train(self):
        pass
