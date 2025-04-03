from abc import ABC, abstractmethod


class IStateManager(ABC):
    @abstractmethod
    def _build_state(self): pass

    @abstractmethod
    def reset(self): pass

    @abstractmethod
    def step(self): pass

    @property
    @abstractmethod
    def is_done(self): pass

    @property
    @abstractmethod
    def current_data(self): pass

    @property
    @abstractmethod
    def state_num(self): pass

    @property
    @abstractmethod
    def data_index(self): pass
