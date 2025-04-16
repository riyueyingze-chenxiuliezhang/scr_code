"""
processor.py
"""
from abc import ABC

import pandas as pd

from ..interface import IDataProcessor
from project.utils.tool.exception_check import *


class BaseDataProcessor(IDataProcessor, ABC):
    """基础处理器（实现公共逻辑）"""
    def __init__(self, config):
        try:
            self._raw_data = pd.read_csv(config.data.data_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"数据文件未找到: {config.data.data_path}") from e

        self._process_features = config.state.process
        self._process_scale = config.state.scale

        if "出口NO2浓度（折算）" in self._process_features:
            self._raw_data["出口NO2浓度（折算）"] = self._raw_data['出口NO2浓度（折算）'].shift()
            self._raw_data = self._raw_data.dropna(subset=['出口NO2浓度（折算）'])

        # 数据滞后条数，同步原始数据
        self._time_lag_data = self._raw_data.shift(-60).dropna()
        self._raw_data = self._raw_data.loc[self._time_lag_data.index]
        self.data_filter()

        type_check(self._process_features, list)
        range_check(type_check(self._process_scale, int), (1, None))

    def data_filter(self):
        features = [
            "煤气压力1热风炉气动阀1前",
            "GGH原烟气侧出口温度",
            "CEM_脱硝入口烟气流量（工况）",
            "入口NO2浓度（折算）",
            "出口NO2浓度（折算）"
        ]
        temp_df = self._raw_data[features].copy()
        mean = temp_df.mean()
        std = temp_df.std()
        upper = mean + 3 * std
        lower = mean - 3 * std

        mask = ((temp_df >= lower) & (temp_df <= upper)).all(axis=1)
        temp_df = temp_df[mask]
        self._raw_data = self._raw_data.loc[temp_df.index].reset_index(drop=True)
        self._time_lag_data = self._time_lag_data.loc[temp_df.index].reset_index(drop=True)

    def get_data(self, index) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Args:
            index: 数据行索引

        Returns:
            返回 当前的状态，时间滞后的状态
        """
        return self._raw_data.iloc[index], self._time_lag_data.iloc[index]

    @property
    def raw_data(self) -> pd.DataFrame:
        return self._raw_data

    @property
    def data_num(self) -> int:
        return self._raw_data.shape[0]

    @property
    def process_features(self):
        return self._process_features

    @property
    def process_scale(self):
        return self._process_scale

    def __repr__(self):
        return f"<DataProcessor using {self._data_path.name}>"
