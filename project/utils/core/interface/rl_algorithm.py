"""
# @Time: 2025/3/26 18:22
# @File: dqn_algorithm.py
"""
from abc import ABC, abstractmethod


class IRLAlgorithm(ABC):
    @abstractmethod
    def update(self): pass

    @abstractmethod
    def train(self): pass

    @abstractmethod
    def save(self, model_path): pass

    @abstractmethod
    def load(self, model_path): pass
