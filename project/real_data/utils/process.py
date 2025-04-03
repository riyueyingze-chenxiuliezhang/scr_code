"""
# @Time: 2025/4/2 10:39
# @File: process.py
"""
import ast
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from project.real_data.config import PROJECT_DIR
from project.utils import MinMaxProcessor, StateSpace, RewardManager
from project.utils.tool.diff import deep_compare_dicts


class Process:
    def __init__(self, config):
        environment = config.environment
        self._config = config

        self._reward_calc = RewardManager(environment.reward)
        self._dp = MinMaxProcessor(config)
        self._state_space = StateSpace(self._dp, environment.state)

        self._temp_data_path = PROJECT_DIR / "config" / self._config.name
        self._df = self._temp_data()

        self._current_index = None

    def _temp_data(self):
        """
        根据配置参数的变动决定是否重新构建四元组
        """
        config_dict = {
            "data_path": self._config.data_path,
            "state": self._config.environment.state.to_dict(),
            "reward": self._config.environment.reward.to_dict()
        }

        temp_config_file = self._temp_data_path / "data_config.yaml"
        temp_data_file = self._temp_data_path / "data.csv"

        if temp_config_file.is_file():
            with open(temp_config_file, "r", encoding="utf-8") as f:
                temp_config = yaml.safe_load(f)
            differences = deep_compare_dicts(config_dict, temp_config)

            # 如果相关的配置参数未变动 获取之前的四元组
            if not differences:
                if temp_data_file.is_file():
                    temp_df = pd.read_csv(temp_data_file)
                    temp_df["state"] = temp_df["state"].apply(ast.literal_eval)
                    temp_df["next_state"] = temp_df["next_state"].apply(ast.literal_eval)
                    return temp_df
                else:
                    return self._build_and_save(temp_data_file)

            # 如果相关的配置参数变动 重新构建四元组
            print(differences)
            return self._build_and_save(temp_data_file)

        # 如果未找到上一次的配置参数
        temp_config_file.parent.mkdir(exist_ok=True, parents=True)
        with open(temp_config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, indent=2, allow_unicode=True, default_flow_style=False)
        return self._build_and_save(temp_data_file)

    def _build_and_save(self, save_path):
        temp_df = self._build_trace()
        temp_df.to_csv(save_path, index=False)
        return temp_df

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
