"""
minmax_processor.py
"""
from typing import Any


from project.utils.core import BaseDataProcessor


class MinMaxProcessor(BaseDataProcessor):
    """最大最小归一化实现"""
    def __init__(self, config):
        super().__init__(config)

        if config.state.get("fixed_max_min", False):
            self._max = config.state.fixed_max_min.max.to_dict()
            self._min = config.state.fixed_max_min.min.to_dict()
        else:
            # 计算最大最小值
            self._max = self._raw_data[self._process_features].max()
            self._min = self._raw_data[self._process_features].min()

    def get_normalized(self, raw_features: dict) -> list[Any]:
        normalized_data = []
        for feature, value in raw_features.items():
            if feature in self._process_features:
                _max = self._max[feature]
                _min = self._min[feature]
                if _max == _min:
                    normalized_data.append(0.0)
                else:
                    normalized_data.append((value - _min) / (_max - _min) * self.process_scale)
            else:
                normalized_data.append(value)
        return normalized_data
