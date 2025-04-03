"""
# @Time: 2025/3/27 10:41
# @File: algorithm.py

整合 Double dqn、Dueling dqn、优先经验回放、NoisyNet
"""
from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from project.utils.core import BaseRLAlgorithm


class Net(nn.Module):
    def __init__(self, _input_size, _output_size, _hidden_layer: List[int] = None):
        super(Net, self).__init__()

        # 检查参数是否合法
        _hidden_layer = _hidden_layer if _hidden_layer else [64]
        assert isinstance(_hidden_layer, List), "隐藏层必须是一个列表"
        assert all(isinstance(n, int) and n > 0 for n in _hidden_layer), "隐藏层维度需为正整数"

        self.all_layers = [_input_size, *_hidden_layer, _output_size]
        self.layers = nn.ModuleList()

        for in_dim, out_dim in zip(self.all_layers[:-2], self.all_layers[1:-1]):
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU()
            ))
        self.layers.append(nn.Linear(self.all_layers[-2], self.all_layers[-1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)


class DQNImpl(BaseRLAlgorithm):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(config)

        self._config = config
        self.device = torch.device(config.device)
        self._hidden_layer = config.hidden_layer
        if config.name == "train":
            self._count = 0
            self._lr = config.lr
            self.epsilon = config.epsilon_start
            self._gamma = config.gamma
            self._update_step = config.update_step
            self._tau = config.tau
            self._dd = config.dd

            self.q_net = Net(state_dim, action_dim, self._hidden_layer).to(self.device)
            self.target_q_net = Net(state_dim, action_dim, self._hidden_layer).to(self.device)
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self._lr)
        elif config.name == "test":
            self.epsilon = config.epsilon

            self.q_net = Net(state_dim, action_dim, self._hidden_layer).to(self.device).eval()
        else:
            raise Exception(f"{config.name} 未期望的参数")

    def update(self):
        """更新神经网络"""
        self._count += 1
        if self._count % self._update_step == 0:
            if self._tau == 0:
                # 硬更新
                self.target_q_net.load_state_dict(self.q_net.state_dict())
            else:
                # 软更新
                for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                    target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param)
            # print(self.epsilon)
        self.epsilon = self._update_epsilon(self._count)

    def train(self, transitions):
        states = torch.from_numpy(np.array(transitions["states"])).type(torch.float).to(self.device)
        actions = torch.tensor(transitions["actions"]).view(-1, 1).to(self.device)
        rewards = torch.tensor(transitions["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.from_numpy(np.array(transitions["next_states"])).type(torch.float).to(self.device)
        dones = torch.tensor(transitions["done"], dtype=torch.float).view(-1, 1).to(self.device)

        q_value = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            if not self._dd:
                next_q_value = self.target_q_net(next_states).max(1)[0].view(-1, 1).detach()
            else:
                """
                1. 用在线网络选择动作
                2. 用目标网络评估 Q 值
                """
                next_actions = self.q_net(next_states).max(1)[1].view(-1, 1)
                next_q_value = self.target_q_net(next_states).gather(1, next_actions).detach()
            target_q_value = next_q_value * self._gamma * (1 - dones) + rewards

        loss = F.mse_loss(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 更新目标网路
        self.update()

        return loss

    def get_q_value(self, state):
        return self.q_net(state)

    def save(self, model_path):
        torch.save({
            "model_state_dict": self.q_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, model_path)

    def load(self, model_path):
        checkpoint = torch.load(model_path)
        if self._config.name == "train":
            self.q_net.load_state_dict(checkpoint['model_state_dict'])
            self.target_q_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif self._config.name == "test":
            self.q_net.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise Exception(f"未期望的参数 {self._config.name}")
