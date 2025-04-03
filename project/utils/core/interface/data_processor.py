"""
data_processor.py
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class IDataProcessor(ABC):
    """数据处理标准接口"""
    @abstractmethod
    def get_normalized(self, raw_features): pass
    """数值归一化方法"""

    @abstractmethod
    def get_data(self, index): pass
    """访问第 index 行数据"""

    @property
    @abstractmethod
    def raw_data(self): pass
    """访问原始数据"""

    @property
    @abstractmethod
    def data_num(self): pass
    """获取原始数据数量"""

