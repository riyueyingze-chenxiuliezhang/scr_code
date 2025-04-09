"""
# @Time: 2025/3/26 14:26
# @File: base_state.py
"""
from collections import deque, OrderedDict

import numpy as np

from project.utils.core import BaseDataProcessor, BaseState


class StateSpace(BaseState):
    def __init__(self, data_processor: BaseDataProcessor, config):
        """
        Args:
            data_processor (BaseDataProcessor): 数据处理类
            config: 配置项 需要包含 valve 中的一些变量
        """
        super().__init__()

        self._dp = data_processor
        self._config = config

        self._prev_action = None
        self._prev_outlet_c = None
        self._curr_action = None
        self._curr_outlet_c = None
        self._current_index = None
        self._history_window = None

        self._state_features = self.flatten_list(self._config.features)

    def _build_state(self):
        row = self.current_data
        features = OrderedDict()
        for key in self._state_features:
            if key == "焦炉煤气阀门开度":
                self._curr_action = row[key]
                features[key] = self._prev_action
            elif key == "出口NO2浓度（折算）":
                self._curr_outlet_c = row[key]
                features[key] = self._prev_outlet_c
            else:
                features[key] = row[key]
        return np.array(self._dp.get_normalized(features), dtype=np.float32)

    def reset(self):
        self._prev_action = self._config.prev_valve
        self._prev_outlet_c = self._config.prev_outlet_c
        self._current_index = self._config.init_data_index
        self._history_window = deque(maxlen=self._config.history_window)
        return self._build_state()

    def step(self):
        self._current_index += 1
        self._prev_action = self._curr_action
        self._prev_outlet_c = self._curr_outlet_c
        return self._build_state()

    @property
    def is_done(self):
        return self._current_index >= self._dp.data_num - 1

    @property
    def current_data(self):
        return self._dp.get_data(self.data_index)

    @property
    def state_num(self):
        return len(self._state_features)

    @property
    def data_index(self):
        return self._current_index

    def flatten_list(self, lst):
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(self.flatten_list(item))
            else:
                result.append(item)
        return result

    def set_prev_valve(self, valve):
        self._prev_action = valve

    def add_history_record(self, record):
        self._history_window.append(record)

    @property
    def get_history_record(self):
        return deque(self._history_window)
