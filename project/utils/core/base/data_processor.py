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
        self._data_path = config.data_path
        try:
            self._raw_data = pd.read_csv(config.data_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"数据文件未找到: {config.data_path}") from e

        self._process_features = config.environment.state.process
        self._process_scale = config.environment.state.scale

        type_check(self._process_features, list)
        range_check(type_check(self._process_scale, int), (1, None))

    def get_data(self, index) -> pd.DataFrame:
        return self._raw_data.iloc[index]

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
