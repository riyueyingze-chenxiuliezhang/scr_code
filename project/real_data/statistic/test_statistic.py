"""
# @Time: 2025/4/1 16:09
# @File: test_statistic.py
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from project.real_data.config import PROJECT_DIR, ConfigLoader

test_result_path = PROJECT_DIR / "result" / "test" / ConfigLoader.name


def test_action_statistic(start, end=None):
    if end is None:
        end = start
    # 为了包含 end episode，这里将 end 加 1
    end += 1

    for episode in range(start, end):
        result_path = test_result_path / str(episode)

        with open(result_path / "action.pkl", "rb") as f:
            action = pickle.load(f)

        with open(result_path / "real_action.pkl", "rb") as f:
            real_action = pickle.load(f)

        df = pd.DataFrame({
            "Index": np.arange(len(action)),
            "action": action,
            "real_action": real_action
        }).melt(id_vars="Index", var_name="Series", value_name="Value")

        fig = px.scatter(
            df,
            x="Index",
            y="Value",
            color="Series",
            symbol="Series",
            title=f"{ConfigLoader.name} Ep {episode} 动作统计"  # 添加图表标题
        )
        fig.show()


def test_action_value_statistic(start, end=None):
    if end is None:
        end = start
    # 为了包含 end episode，这里将 end 加 1
    end += 1

    avg_action_value = []
    for episode in range(start, end):
        result_path = test_result_path / str(episode)

        with open(result_path / "action_value.pkl", "rb") as f:
            action_value = pickle.load(f)
        avg_action_value.append(sum(action_value) / len(action_value))

    plt.plot(avg_action_value)
    plt.xlabel("Training epochs")
    plt.ylabel("Average action value (Q)")
    plt.show()


if __name__ == '__main__':
    test_action_statistic(19, 19)
    # test_action_value_statistic(0, 19)
