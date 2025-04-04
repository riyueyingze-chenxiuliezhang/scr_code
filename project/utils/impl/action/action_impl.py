"""
# @Time: 2025/3/26 14:11
# @File: base_action.py

基础 action
"""
import numpy as np

from project.utils.core import BaseAction


class ActionSpace(BaseAction):
    def __init__(self, config):
        super().__init__()

        self.actions = np.arange(
            config.action_min,
            config.action_max + config.action_step,
            config.action_step
        )

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        return self.actions[index]

    @property
    def action_num(self):
        return len(self.actions)

    def transform_to_index(self, action):
        mask = np.isclose(self.actions, action)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            raise ValueError(f"Action {action} is not valid. Available actions: {self.actions.tolist()}")
        return int(indices[0])
