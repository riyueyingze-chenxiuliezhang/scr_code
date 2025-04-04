"""
# @Time: 2025/3/26 13:52
# @File: action.py
"""
from abc import ABC

from ..interface import IActionManager


class BaseAction(IActionManager, ABC):
    def __init__(self): pass
