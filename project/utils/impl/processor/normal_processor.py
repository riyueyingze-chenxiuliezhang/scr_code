"""
# @Time: 2025/3/26 13:33
# @File: normal_processor.py
"""
from typing import Any

import numpy as np

from project.utils.core import BaseDataProcessor


class NormalProcessor(BaseDataProcessor):
    """标准化实现"""
    def __init__(self, config):
        """
        Args:
            config: 配置项 需要包含数据路径（data_path)
        """
        super().__init__(config)

        # 计算均值和方差
        self._means = self.raw_data[self._process_features].mean()
        self._stds = self.raw_data[self._process_features].std()

    def get_normalized(self, raw_features: dict) -> list[Any]:
        normalized_data = []
        for feature, value in raw_features.items():
            if feature in self.process_features:
                mean = self._means[feature]
                std = self._stds[feature]
                if np.isclose(std, 0):
                    normalized_data.append(0.0)
                else:
                    normalized_data.append((value - mean) / std * self.process_scale)
            else:
                normalized_data.append(value)
        return normalized_data
