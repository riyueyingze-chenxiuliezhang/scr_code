import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from project.real_data.config import PROJECT_DIR, ConfigLoader


def test_statistic():
    result_path = PROJECT_DIR / "result" / "test" / ConfigLoader.name

    with open(result_path / "real_target.pkl", "rb") as f:
        real_target = pickle.load(f)

    with open(result_path / "real_outlet_c.pkl", "rb") as f:
        real_outlet_c = pickle.load(f)

    with open(result_path / "model_predict_outlet_c.pkl", "rb") as f:
        predict_outlet_c = pickle.load(f)

    # 数据校验
    assert len(real_target) == len(real_outlet_c) == len(predict_outlet_c), "数据长度不一致"

    start_index = 0

    real_target = real_target[start_index:start_index + 5_0000]
    real_outlet_c = real_outlet_c[start_index:start_index + 5_0000]
    predict_outlet_c = predict_outlet_c[start_index:start_index + 5_0000]

    # 创建绘图数据框架
    df = pd.DataFrame({
        'Index': np.arange(len(real_target)),  # x轴索引
        '指标': real_target,
        '真实出口浓度': real_outlet_c,
        '预测出口浓度': predict_outlet_c
    }).melt(id_vars='Index',
            var_name='Series',
            value_name='Value')

    # 创建散点图
    fig = px.scatter(
        df,
        x='Index',
        y='Value',
        color='Series',
        symbol='Series',  # 用不同形状区分序列
        opacity=0.7,
        title='Concentration Comparison',
        labels={'Index': 'Time Step', 'Value': 'Concentration (ppm)'},
        template='plotly_white'
    )

    # 格式优化
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        # hovermode='x unified'
    )

    fig.show()


def data_statistic():
    data_path = r"C:\Users\admi\Desktop\aaa\data\process\25_0322-0327_process.csv"
    df = pd.read_csv(data_path)

    # 确保时间列是datetime类型
    df['时间'] = pd.to_datetime(df['时间'])
    temp_df = df[[
        "时间",
        "出口NO2浓度（折算）",
        "焦炉煤气阀门开度",
        "指标",
    ]]
    temp_df = temp_df.melt(id_vars='时间',
                           var_name='Series',
                           value_name='Value')

    # 创建双变量散点图
    fig = px.scatter(
        temp_df,
        x="时间",
        y="Value",  # 主Y轴数据
        color='Series',
        symbol='Series',  # 用不同形状区分序列
        title="出口NO2浓度与指标时序分析",
        labels={"出口NO2浓度（折算）": "NO2浓度（mg/m³）"}
    )

    # 优化图表布局
    fig.update_layout(
        xaxis_title="时间",
        yaxis_title="数值",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    fig.show()


if __name__ == '__main__':
    data_statistic()
    # for i in range(5):
    #     test_action(i)
    # test_action(4)
