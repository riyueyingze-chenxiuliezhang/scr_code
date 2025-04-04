"""
# @Time: 2025/3/26 20:09
# @File: rl_agent.py
"""
from abc import ABC, abstractmethod


class IRLAgent(ABC):

    @abstractmethod
    def select_action(self, state): pass

    @abstractmethod
    def update_network(self): pass

    @abstractmethod
    def save_network(self, dir_path, epoch): pass

    @abstractmethod
    def load_network(self, dir_path, _epoch): pass
