"""
# @Time: 2025/3/26 14:09
# @File: env.py
"""
from project.template.config import ActionSpace, RewardManager, NormalProcessor, StateSpace, ConfigLoader


class SCREnv:
    def __init__(self):
        self._action_space = ActionSpace(ConfigLoader.environment)
        self._reward_calc = RewardManager(ConfigLoader.environment)
        self._dp = NormalProcessor(ConfigLoader)
        self._state_space = StateSpace(self._dp, ConfigLoader.environment)

    def reset(self):
        self._state_space.reset()
        return self._state_space.build_state()

    def step(self):
        self._state_space.step()
        reward = self._reward_calc()
        others = ()
        return self._state_space.build_state(), reward, self._state_space.is_done, others

    @property
    def data_num(self):
        return self._dp.data_num

    @property
    def action_dim(self):
        return self._action_space.action_num

    @property
    def state_dim(self):
        return self._state_space.state_num
