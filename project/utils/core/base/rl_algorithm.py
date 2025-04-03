"""
# @Time: 2025/3/26 20:11
# @File: algorithm.py
"""
from abc import ABC

import numpy as np

from ..interface import IRLAlgorithm


class BaseRLAlgorithm(IRLAlgorithm, ABC):
    def __init__(self, config):
        if config.device not in ["cuda", "cpu"]:
            raise Exception(f"配置项异常 device 为 {config.device} 不在 ['cuda', 'cpu']中")

        if config.name == "train":
            self._min_epsilon = config.epsilon_end
            self._epsilon_delay = config.epsilon_delay

    def _update_epsilon(self, count):
        self._min_epsilon + (1 - self._min_epsilon) * np.exp(-1 * count / self._epsilon_delay)
