"""
# @Time: 2025/3/26 14:09
# @File: env.py
"""
import numpy as np

from .process import Process
from project.real_data.config import ConfigLoader
from project.utils import ActionSpace


class TrainSCREnv:
    def __init__(self):
        environment = ConfigLoader.environment

        self._action_space = ActionSpace(environment.action)
        self._process = Process(ConfigLoader)

    def reset(self):
        self._process.reset()

    def step(self):
        """ 使用真实数据，不接收外部动作 """
        current_data = self._process.step()

        state = np.array(current_data['state'])
        action = self._action_space.transform_to_index(current_data['action'])  # 将真实动作转化为索引
        reward = current_data['reward']
        next_state = np.array(current_data['next_state'])
        done = current_data['done']

        return state, action, reward, next_state, done, self._process.is_done

    @property
    def data_num(self):
        return self._process.data_num

    @property
    def action_dim(self):
        return self._action_space.action_num

    @property
    def state_dim(self):
        return self._process.state_num
