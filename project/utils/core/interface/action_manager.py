from abc import ABC, abstractmethod


class IActionManager(ABC):
    @abstractmethod
    def __len__(self): pass

    @abstractmethod
    def __getitem__(self, index): pass

    @property
    @abstractmethod
    def action_num(self): pass
