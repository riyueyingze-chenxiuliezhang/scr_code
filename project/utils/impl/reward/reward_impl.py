"""
# @Time: 2025/3/26 14:42
# @File: base_reward.py
"""
import numpy as np

from project.utils.core import BaseReward


class RewardManager(BaseReward):
    def __init__(self, config):
        """
        Args:
            config: 配置项 需要包含 valve 中的一些初始化变量
        """
        super().__init__(config)

    def __call__(self, *args, **kwargs):
        """
        计算组合奖励

        Args:
            curr_outlet_c:      当前的真实/预测的出口浓度
            target_outlet_c:    目标出口浓度
            curr_action:        当前选择的动作
            prev_action:        上一次的动作
        """
        return self.calculate(*args, **kwargs)

    def calculate(self, curr_outlet_c, target_outlet_c, curr_action=None, prev_action=None):
        r1 = self._calc_emission_reward(curr_outlet_c, target_outlet_c, self.sigma, self.k)
        # r2 = self._calc_action_penalty(curr_action, prev_action, self.c)
        # return r1 * self.emission_weight + r2 * self.valve_weight
        # return r1 - r2  # 减少超参数数量
        return r1

    def _calc_emission_reward(self, current, target, sigma=3, k=1):
        dis = current - target
        if current <= target:
            return np.exp(-dis**2 / (2 * sigma**2)) * self.pos_scale
        return np.clip(- k * dis**2, self.neg_clip, 0)

    @classmethod
    def _calc_action_penalty(cls, curr_action, prev_action, c=0.05):
        action_change = abs(curr_action - prev_action)
        return c * action_change
