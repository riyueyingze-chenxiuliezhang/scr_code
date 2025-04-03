"""
# @Time: 2025/3/26 13:38
# @File: state.py
"""
from abc import ABC

from ..interface import IStateManager


class BaseState(IStateManager, ABC):
    def __init__(self): pass
