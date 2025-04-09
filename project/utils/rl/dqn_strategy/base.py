"""
# @Time: 2025/4/8 18:43
# @File: base.py
"""
import numpy as np
import torch
import torch.nn.functional as F

from .strategy import BaseStrategy
from ..dqn_impl import DQNImpl


@DQNImpl.register_strategy("base")
class BaseDqn(BaseStrategy):
    def __init__(self, context):
        self._context = context

    def sample(self):
        s, a, r, ns, d = self._context.replay_memory.sample(self._context.batch_size)

        states = torch.from_numpy(np.array(s)).type(torch.float).to(self._context.device)
        actions = torch.tensor(a).view(-1, 1).to(self._context.device)
        rewards = torch.tensor(r, dtype=torch.float).view(-1, 1).to(self._context.device)
        next_states = torch.from_numpy(np.array(ns)).type(torch.float).to(self._context.device)
        dones = torch.tensor(d, dtype=torch.float).view(-1, 1).to(self._context.device)

        return states, actions, rewards, next_states, dones

    def predict(self, states, actions, rewards, next_states, dones):
        q_value = self._context.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_value = self._context.target_q_net(next_states).max(1)[0].view(-1, 1)
            target_q_value = next_q_value * self._context.gamma * (1 - dones) + rewards

        return q_value, target_q_value

    def update(self):
        self._context.count += 1
        if self._context.count % self._context.update_step == 0:
            self._context.target_q_net.load_state_dict(self._context.q_net.state_dict())
        self._context.epsilon = self._context.update_epsilon(self._context.count)

    def train(self):
        experiences = self._context.sample()
        q_value, target_q_value = self._context.predict(*experiences)

        loss = F.mse_loss(q_value, target_q_value)
        self._context.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self._context.optimizer.step()

        # 更新目标网路
        self.update()

        return loss
