"""
# @Time: 2025/3/26 20:11
# @File: agent.py
"""
from abc import ABC

from ..interface import IRLAgent


class BaseRLAgent(IRLAgent, ABC):
    def __init__(self): pass
