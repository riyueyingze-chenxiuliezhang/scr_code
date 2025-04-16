"""
# @Time: 2025/3/27 10:41
# @File: algorithm.py

整合 Double dqn、Dueling dqn、优先经验回放、NoisyNet
"""

import torch

from project.utils.core import BaseRLAlgorithm
from ..exp_replay import replay_memory
from ..net_impl import dqn_net


class DQNImpl(BaseRLAlgorithm):
    _strategies = {}  # 策略注册表

    def __init__(self, state_dim, action_dim, config):
        super().__init__(config)

        net_config = {
            "input_size": state_dim,
            "output_size": action_dim,
            "hidden_layer": config.hidden_layer,
            "net_name": config.net_name
        }
        sample_name = "pri" if config.get("sample_method", "base") == "pri" else "base"
        predict_name = "dd" if config.get("dd", False) else "base"
        update_name = "tau" if config.get("tau", 0) != 0 else "base"
        train_name = sample_name

        if self.config_name == "train":
            self.count = 0
            self._lr = config.lr
            self.epsilon = config.epsilon_start
            self.gamma = config.gamma
            self.update_step = config.update_step
            self.batch_size = config.batch_size
            self.tau = config.tau
            self.dd = config.dd
            self.replay_memory = replay_memory(config.capacity, sample_name)

            self.q_net = dqn_net(**net_config).to(self.device)
            self.target_q_net = dqn_net(**net_config).to(self.device)
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self._lr)
        elif self.config_name == "test":
            self.epsilon = config.epsilon

            self.q_net = dqn_net(**net_config).to(self.device).eval()
        else:
            raise Exception(f"{self.config_name} 未期望的参数")

        self._sample_func = self._strategies[sample_name](self).sample
        self._predict_func = self._strategies[predict_name](self).predict
        self._update_func = self._strategies[update_name](self).update
        self._train_func = self._strategies[train_name](self).train

    def sample(self):
        """ 从经验池中采样 """
        return self._sample_func()

    def predict(self, states, actions, rewards, next_states, dones):
        """ 预测 q 值 和 next_q 值 """
        return self._predict_func(states, actions, rewards, next_states, dones)

    def update(self):
        """ 更新神经网络 """
        self._update_func()

    def train(self):
        return self._train_func()

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_memory.add(state, action, reward, next_state, done)

    def get_q_value(self, state):
        return self.q_net(state)

    def save(self, model_path):
        torch.save({
            "model_state_dict": self.q_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, model_path)

    def load(self, model_file):
        checkpoint = torch.load(model_file)
        if self.config_name == "train":
            self.q_net.load_state_dict(checkpoint['model_state_dict'])
            self.target_q_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif self.config_name == "test":
            self.q_net.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise Exception(f"未期望的参数 {self.config_name}")

    @classmethod
    def register_strategy(cls, name):
        def decorator(strategy_cls):
            cls._strategies[name] = strategy_cls
            return strategy_cls

        return decorator

    @property
    def replay_memory_len(self):
        return len(self.replay_memory)
