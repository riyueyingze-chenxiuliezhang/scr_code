"""
# @Time: 2025/3/27 13:58
# @File: no_processor.py
"""
from typing import Any


from project.utils.core import BaseDataProcessor


class NoProcessor(BaseDataProcessor):
    """最大最小归一化实现"""
    def __init__(self, config):
        super().__init__(config)

        # 计算最大最小值
        self._max = self._raw_data[self._process_features].max()
        self._min = self._raw_data[self._process_features].min()

    def get_normalized(self, raw_features: dict) -> list[Any]:
        normalized_data = []
        for feature, value in raw_features.items():
            normalized_data.append(value)
        return normalized_data
