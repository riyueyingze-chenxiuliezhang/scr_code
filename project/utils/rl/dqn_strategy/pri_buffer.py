"""
# @Time: 2025/4/8 19:29
# @File: pro_pri.py
"""
import numpy as np
import torch
import torch.nn.functional as F

from .strategy import BaseStrategy
from ..dqn_impl import DQNImpl


@DQNImpl.register_strategy("pri")
class PriBuffer(BaseStrategy):
    def __init__(self, context):
        self._context = context

    def sample(self):
        transitions, indices, weights = self._context.replay_memory.sample(self._context.batch_size)
        s, a, r, ns, d = transitions

        states = torch.from_numpy(np.array(s)).type(torch.float).to(self._context.device)
        actions = torch.tensor(a).view(-1, 1).to(self._context.device)
        rewards = torch.tensor(r, dtype=torch.float).view(-1, 1).to(self._context.device)
        next_states = torch.from_numpy(np.array(ns)).type(torch.float).to(self._context.device)
        dones = torch.tensor(d, dtype=torch.float).view(-1, 1).to(self._context.device)

        return (states, actions, rewards, next_states, dones), indices, weights

    def predict(self, states, actions, rewards, next_states, dones):
        pass

    def update(self):
        pass

    def train(self):
        experiences, indices, weights = self._context.sample()
        q_value, target_q_value = self._context.predict(*experiences)

        td_errors = torch.abs(target_q_value - q_value).detach().cpu().numpy()
        self._context.replay_memory.update_priorities(indices, td_errors)

        loss = F.mse_loss(q_value, target_q_value)
        loss = (loss * torch.tensor(weights).to(self._context.device)).mean()
        self._context.optimizer.zero_grad()
        loss.backward()
        self._context.optimizer.step()

        # 更新目标网路
        self._context.update()

        return loss
