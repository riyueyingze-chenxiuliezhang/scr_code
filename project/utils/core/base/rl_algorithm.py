"""
# @Time: 2025/3/26 20:11
# @File: algorithm.py
"""
from abc import ABC

import numpy as np
import torch

from ..interface import IRLAlgorithm


class BaseRLAlgorithm(IRLAlgorithm, ABC):
    def __init__(self, config):
        if config.device not in ["cuda", "cpu"]:
            raise Exception(f"配置项异常 device 为 {config.device} 不在 ['cuda', 'cpu']中")

        self.device = torch.device(config.device)
        self.config_name = config.name

        if self.config_name == "train":
            self._min_epsilon = config.epsilon_end
            self._epsilon_delay = config.epsilon_delay

    def update_epsilon(self, count):
        return self._min_epsilon + (1 - self._min_epsilon) * np.exp(-1 * count / self._epsilon_delay)
