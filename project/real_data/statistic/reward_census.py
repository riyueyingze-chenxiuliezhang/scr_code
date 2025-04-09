"""
# @Time: 2025/3/31 20:46
# @File: s_reward.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from project.utils import RewardManager, DataRecorder
from project.utils.tool.draw_figure import draw_bar
from project.real_data.config import ConfigLoader

save_path = Path(__file__).parent / "result" / "reward"


def record(data_path=None):
    r = RewardManager(ConfigLoader.environment.reward)
    data_record = DataRecorder(save_path, file_fmt="pkl")

    data = pd.read_csv(ConfigLoader.data_path if not data_path else data_path)
    progress_bar = tqdm(total=data.shape[0])

    for _, row in data.iterrows():
        action = row['焦炉煤气阀门开度']
        target = row['指标']
        outlet_c = row['出口NO2浓度（折算）']
        reward = r(outlet_c, target, 0, 0)
        data_record.add_data("reward", reward)
        data_record.add_data("action", action)
        data_record.add_data("target", target)
        data_record.add_data("outlet_c", outlet_c)
        progress_bar.update(1)

    data_record.flush()


def plot():
    with open(save_path / "action.pkl", "rb") as f:
        action = pickle.load(f)

    with open(save_path / "reward.pkl", "rb") as f:
        reward = pickle.load(f)

    with open(save_path / "target.pkl", "rb") as f:
        target = pickle.load(f)

    with open(save_path / "outlet_c.pkl", "rb") as f:
        outlet_c = pickle.load(f)

    df = pd.DataFrame({
        "index": np.arange(len(action)),
        "action": action,
        "reward": reward,
        "target": target,
        "outlet_c": outlet_c
    }).melt(id_vars="index",
            var_name="series",
            value_name="value")

    fig = px.scatter(
        df,
        x="index",
        y="value",
        color="series",
        symbol="series"
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


def statistic():
    with open(save_path / "action.pkl", "rb") as f:
        action = pickle.load(f)

    with open(save_path / "reward.pkl", "rb") as f:
        reward = pickle.load(f)

    with open(save_path / "target.pkl", "rb") as f:
        target = pickle.load(f)

    df = pd.DataFrame({
        "index": np.arange(len(action)),
        "action": action,
        "reward": reward,
        "target": target
    })

    # 统计条件
    conditions = {
        "raw": df.shape[0],
        "reward < 0": (df['reward'] < 0).sum(),
        "action <= 14": (df['action'] < 14).sum(),
        "reward < 0 and \n action <= 14": ((df['reward'] < 0) & (df['action'] <= 14)).sum(),
    }
    categories = list(conditions.keys())
    values = list(conditions.values())
    draw_bar(categories, values)
    # low_reward_count = condition.sum()
    # print(f"统计结果：{low_reward_count}")


if __name__ == '__main__':
    # record(r"C:\Users\admi\Desktop\aaa\data\process\25_0204-0226_single_filter.csv")
    record()
    # statistic()
    plot()
