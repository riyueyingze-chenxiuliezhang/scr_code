"""
# @Time: 2025/4/2 10:39
# @File: process.py
"""
import pandas as pd
from tqdm import tqdm

from project.utils import NormalProcessor, StateSpace, RewardManager


class Process:
    def __init__(self, config):
        environment = config.environment
        self._config = config

        self._reward_calc = RewardManager(environment.reward)
        self._dp = NormalProcessor(config)
        self._state_space = StateSpace(self._dp, environment.state)

        self._df = self._build_trace()

        self._current_index = None

    def _build_trace(self):
        progress_bar = tqdm(total=self.data_num, desc="构建轨迹")

        transitions = []
        state = self._state_space.reset()

        while True:
            data_row = self._state_space.current_data
            real_outlet_c = data_row['出口NO2浓度（折算）']
            target_outlet_c = data_row['指标']

            action = data_row['焦炉煤气阀门开度']
            reward = self._reward_calc(real_outlet_c, target_outlet_c)
            next_state = self._state_space.step()
            done = self._state_space.is_done

            transitions.append({
                "state": state.tolist(),  # 状态
                "action": action,  # 动作
                "reward": reward,  # 奖励
                "next_state": next_state.tolist(),  # 下一状态
                "done": done  # 是否终止
            })

            state = next_state
            progress_bar.update(1)

            if done:
                break

        df = pd.DataFrame.from_records(transitions).astype({
            "action": "int64",
            "reward": "float32",
            "done": "bool"
        })
        progress_bar.close()

        return df

    def reset(self):
        self._current_index = -1
        self._df = self._df.sample(frac=1).reset_index(drop=True)

    def step(self):
        self._current_index += 1
        return self._current_data

    @property
    def _current_data(self):
        return self._df.iloc[self._current_index]

    @property
    def is_done(self):
        return self._current_index >= self._df.shape[0] - 1

    @property
    def data_num(self):
        return self._dp.data_num

    @property
    def state_num(self):
        return self._state_space.state_num
