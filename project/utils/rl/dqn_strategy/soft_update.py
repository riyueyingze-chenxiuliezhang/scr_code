"""
# @Time: 2025/4/8 18:48
# @File: soft_update.py
"""
from .strategy import BaseStrategy
from ..dqn_impl import DQNImpl


@DQNImpl.register_strategy("tau")
class SoftUpdate(BaseStrategy):
    def __init__(self, context):
        self._context = context

    def sample(self):
        pass

    def predict(self, states, actions, rewards, next_states, dones):
        pass

    def update(self):
        """更新神经网络"""
        self._context.count += 1
        if self._context.count % self._context.update_step == 0:
            # 软更新
            for target_param, param in zip(self._context.target_q_net.parameters(), self._context.q_net.parameters()):
                target_param.data.copy_(self._context.tau * param.data + (1 - self._context.tau) * target_param)
        self._context.epsilon = self._context.update_epsilon(self._context.count)

    def train(self):
        pass
