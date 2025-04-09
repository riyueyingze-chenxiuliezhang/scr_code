"""
# @Time: 2025/3/27 10:41
# @File: agent.py
"""
from pathlib import Path

from project.utils.core import BaseRLAgent


class DQNAgentImpl(BaseRLAgent):
    def __init__(self):
        super().__init__()

    def select_action(self, _state):
        pass

    def update_network(self):
        pass

    def add_experience(self, state, action, reward, next_state, done):
        pass

    def save_network(self, dir_path: Path, epoch):
        pass

    def load_network(self, dir_path: Path, _epoch):
        pass
