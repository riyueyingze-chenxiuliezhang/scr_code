"""
# @Time: 2025/3/26 14:09
# @File: env.py
"""
from collections import OrderedDict

import numpy as np

from project.real_data.config import ConfigLoader
from project.utils import (ActionSpace,
                           RewardManager)
from project.utils.factory.processor import data_processor


class TestSCREnv:
    """
    测试环境不计算奖励
    """
    def __init__(self):
        # environment = ConfigLoader.environment
        #
        # self._action_space = ActionSpace(environment.action)
        # self._reward_calc = RewardManager(environment.reward)
        # self._dp = data_processor(environment)
        # self._state_space = StateSpace(self._dp, environment.state)

        # 直接读取 csv 文件
        self._environment = ConfigLoader.environment
        self._action_space = ActionSpace(self._environment.action)
        self._reward_calc = RewardManager(self._environment.reward)
        self._dp = data_processor(self._environment)
        self._data = self._dp.raw_data

        self._process_features = self._environment.state.process
        self._state_features = self._flatten_list(self._environment.state.features)

        self._data_index = None
        self._prev_action = None
        self._prev_outlet_c = None
        self._curr_action = None
        self._curr_outlet_c = None

    def _flatten_list(self, lst):
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(self._flatten_list(item))
            else:
                result.append(item)
        return result

    def _get_normalized(self, raw_features: dict):
        # normalized_data = []
        # for feature, value in raw_features.items():
        #     if feature in self._process_features:
        #         _max = self._max[feature]
        #         _min = self._min[feature]
        #         if _max == _min:
        #             normalized_data.append(0.0)
        #         else:
        #             normalized_data.append((value - _min) / (_max - _min) * self._environment.state.scale)
        #     else:
        #         normalized_data.append(value)
        return self._dp.get_normalized(raw_features)

    def _build_state(self):
        row = self.current_data
        features = OrderedDict()
        for key in self._state_features:
            # if key == "焦炉煤气阀门开度":
            #     self._curr_action = row[key]
            #     features[key] = self._prev_action
            # elif key == "出口NO2浓度（折算）":
            #     self._curr_outlet_c = row[key]
            #     features[key] = self._prev_outlet_c
            # else:
            #     features[key] = row[key]
            features[key] = row[key]
        return np.array(self._get_normalized(features), dtype=np.float32)

    def reset(self):
        self._data_index = -1
        # self._data_index = 1
        # self._prev_action = self._data.iloc[self._data_index - 1]['焦炉煤气阀门开度']
        # self._prev_outlet_c = self._data.iloc[self._data_index - 1]['出口NO2浓度（折算）']
        information = {
            "raw_data_row": self.current_data
        }
        return self._build_state(), information

    def step(self):
        """
        使用真实数据，不接收外部动作
        """
        # 索引先加 1 拿到下一条数据
        self._data_index += 1

        self._prev_action = self._curr_action
        self._prev_outlet_c = self._curr_outlet_c

        # data_row = self.current_data
        # real_outlet_c = data_row['出口NO2浓度（折算）']
        # target_outlet_c = data_row['目标浓度']
        # reward = self._reward_calc(real_outlet_c, target_outlet_c)

        information = {
            "raw_data_row": self.current_data
        }

        return self._build_state(), None, self.is_done, information

    @property
    def current_data(self):
        return self._data.iloc[self._data_index]

    @property
    def data_num(self):
        return self._data.shape[0]

    @property
    def action_dim(self):
        return self._action_space.action_num

    @property
    def state_dim(self):
        return len(self._state_features)

    @property
    def is_done(self):
        return self._data_index >= self.data_num - 1
