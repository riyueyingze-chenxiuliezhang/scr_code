"""
# @Time: 2025/4/1 16:09
# @File: train_statistic.py
"""
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt

from project.real_data.config import STAT_SAVE_DIR


# def train_statistic(episode):
#     result_path = STAT_SAVE_DIR / str(episode)
#
#     with open(result_path / "result_reward.pkl", "rb") as f:
#         result_reward = pickle.load(f)
#
#     with open(result_path / "result_loss.pkl", "rb") as f:
#         result_loss = pickle.load(f)[:]
#
#     plt.figure()
#     plt.plot(range(len(result_reward)), result_reward)
#     plt.title('reward')
#     plt.xlabel('train episode')
#     plt.ylabel('average reward per episode')
#
#     plt.figure()
#     plt.plot(range(len(result_loss)), result_loss)
#     plt.title('loss')
#     plt.xlabel('train episode')
#     plt.ylabel('average loss per episode')
#
#     plt.show()


def train_statistic(episode_range, n_cols=5, train_path=None, name="loss"):
    # 计算总episode数和需要的行数
    num_episodes = episode_range[1] - episode_range[0] + 1
    n_rows = math.ceil(num_episodes / n_cols)

    # 创建子图网格，确保axes始终是二维数组
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(8, 6),
        squeeze=False  # 保证返回二维数组
    )

    index = None
    # 遍历每个episode绘制损失曲线
    for index, episode in enumerate(range(episode_range[0], episode_range[1] + 1)):
        if not train_path:
            episode_path = STAT_SAVE_DIR / str(episode)
        else:
            episode_path = train_path / str(episode)

        try:
            # 加载损失数据
            with open(episode_path / f"result_{name}.pkl", "rb") as f:
                loss_data = pickle.load(f)

            # 计算子图位置
            row = index // n_cols
            col = index % n_cols

            # 绘制曲线并设置标题
            axes[row, col].plot(loss_data)
            axes[row, col].set_title(f"Ep {episode}", fontsize=8)
            axes[row, col].tick_params(axis='both', labelsize=6)

        except FileNotFoundError:
            print(f"Warning: Missing data for episode {episode}")
            continue

    # 隐藏多余的空子图
    for j in range(index + 1, n_rows * n_cols):
        row = j // n_cols
        col = j % n_cols
        axes[row, col].axis("off")

    # 调整布局并增加整体标题
    plt.tight_layout()
    fig.suptitle("Training Loss Across Episodes", y=1.02, fontsize=12)
    plt.show()


def train_all_statistic(episode_range, name="loss", train_path=None):
    # 计算总episode数和需要的行数
    num_episodes = episode_range[1] - episode_range[0] + 1

    loss_all_data = []
    for episode in range(episode_range[0], episode_range[1] + 1):
        if not train_path:
            episode_path = STAT_SAVE_DIR / str(episode)
        else:
            episode_path = train_path / str(episode)
        with open(episode_path / f"result_{name}.pkl", "rb") as f:
            loss_data = pickle.load(f)
        loss_all_data.extend(loss_data)

    df = pd.DataFrame({
        "loss": loss_all_data
    })

    fig = px.line(
        df,
        title=f"Ep {episode_range[0]} - {episode_range[1]} 的 {name} 图"
    )

    fig.show()


if __name__ == '__main__':
    # train_statistic(9)
    # train_statistic((0, 10), 3, "loss")
    train_all_statistic((1, 19), "loss")
