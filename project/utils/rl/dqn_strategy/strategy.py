"""
# @Time: 2025/4/8 20:21
# @File: strategy.py
"""


class BaseStrategy:

    def sample(self):
        raise NotImplementedError

    def predict(self, states, actions, rewards, next_states, dones):
        raise NotImplementedError

    def update(self):
        """更新神经网络"""
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
