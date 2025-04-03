"""
# @Time: 2025/3/26 13:53
# @File: reward.py
"""
from abc import ABC

from ..interface import IRewardCalculator
from project.utils.tool.exception_check import *


class BaseReward(IRewardCalculator, ABC):
    def __init__(self, config):
        self._pos_scale = config.pos_scale
        self._neg_clip = config.neg_clip
        self._sigma = config.sigma
        self._k = config.k
        self._c = config.c

        range_check(type_check(self._pos_scale, int), (1, None))
        range_check(type_check(self._neg_clip, int), (None, 0))
        range_check(type_check(self._sigma, int), (1, 10))
        range_check(type_check(self._k, int), (1, 10))
        range_check(type_check(self._c, int), (0, None))

    @property
    def pos_scale(self):
        return self._pos_scale

    @property
    def neg_clip(self):
        return self._neg_clip

    @property
    def sigma(self):
        return self._sigma

    @property
    def k(self):
        return self._k

    @property
    def c(self):
        return self._c
