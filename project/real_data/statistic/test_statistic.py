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

        with open(result_path / "real_target.pkl", "rb") as f:
            real_target = pickle.load(f)

        with open(result_path / "real_outlet_c.pkl", "rb") as f:
            real_outlet_c = pickle.load(f)

        df = pd.DataFrame({
            "action": action,
            "real_action": real_action,
            "real_target": real_target,
            "real_outlet_c": real_outlet_c
        })

        condition_count = (((df['real_outlet_c'] > df['real_target']) & (df['real_action'] > df['action'])) |
                           ((df['real_outlet_c'] < df['real_target']) & (df['real_action'] < df['action'])))
        count = condition_count.sum()
        print(count)


def test_action_plot(start, end=None):
    if end is None:
        end = start
    # 为了包含 end episode，这里将 end 加 1
    end += 1

    for episode in range(start, end):
        result_path = test_result_path / str(episode)

        with open(result_path / "time.pkl", "rb") as f:
            time = pickle.load(f)

        with open(result_path / "action.pkl", "rb") as f:
            action = pickle.load(f)

        with open(result_path / "real_action.pkl", "rb") as f:
            real_action = pickle.load(f)

        with open(result_path / "real_target.pkl", "rb") as f:
            real_target = pickle.load(f)

        with open(result_path / "real_outlet_c.pkl", "rb") as f:
            real_outlet_c = pickle.load(f)

        df = pd.DataFrame({
            "Index": pd.to_datetime(time),
            "action": action,
            "real_action": real_action,
            "real_target": real_target,
            "real_outlet_c": real_outlet_c
        })[:50000].melt(id_vars="Index", var_name="Series", value_name="Value")

        fig = px.scatter(
            df,
            x="Index",
            y="Value",
            color="Series",
            symbol="Series",
            title=f"{ConfigLoader.name} Ep {episode} 动作统计"  # 添加图表标题
        )

        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )
        fig.show()


def test_avg_value_statistic(start, end=None, name="action_value"):
    if end is None:
        end = start
    # 为了包含 end episode，这里将 end 加 1
    end += 1

    avg_value = []
    for episode in range(start, end):
        result_path = test_result_path / str(episode)

        with open(result_path / f"{name}.pkl", "rb") as f:
            action_value = pickle.load(f)
        avg_value.append(sum(action_value) / len(action_value))

    # print(avg_value)
    plt.plot(avg_value)
    plt.xlabel("Training epochs")
    plt.ylabel(f"Average {' '.join(name.split('_'))}")
    plt.show()


if __name__ == '__main__':
    # test_action_statistic(0, 19)
    test_action_plot(19)
    # test_avg_value_statistic(0, 19, "action")
    # test_avg_value_statistic(0, 19, "action_value")
    # test_avg_value_statistic(0, 10, "real_action")
