"""
# @Time: 2025/3/26 14:09
# @File: env.py
"""
from project.real_data.config import ConfigLoader
from project.utils import (ActionSpace,
                           RewardManager,
                           NormalProcessor, StateSpace)


class TestSCREnv:
    def __init__(self):
        environment = ConfigLoader.environment

        self._action_space = ActionSpace(environment.action)
        self._reward_calc = RewardManager(environment.reward)
        self._dp = NormalProcessor(ConfigLoader)
        self._state_space = StateSpace(self._dp, environment.state)

    def reset(self):
        return self._state_space.reset()

    def step(self):
        """ 使用真实数据，不接收外部动作 """
        data_row = self._state_space.current_data

        real_curr_valve = data_row['焦炉煤气阀门开度']
        real_outlet_c = data_row['出口NO2浓度（折算）']
        target_outlet_c = data_row['指标']

        reward = self._reward_calc(real_outlet_c, target_outlet_c)
        information = {
            "action": self._action_space.transform_to_index(real_curr_valve),
            "raw_data_row": data_row,
            "data_index": self._state_space.data_index
        }

        return self._state_space.step(), reward, self._state_space.is_done, information

    @property
    def data_num(self):
        return self._dp.data_num

    @property
    def action_dim(self):
        return self._action_space.action_num

    @property
    def state_dim(self):
        return self._state_space.state_num
