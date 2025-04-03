from abc import ABC, abstractmethod


class IRewardCalculator(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs): pass

    @abstractmethod
    def calculate(self, current, target, valve_change, history): pass
