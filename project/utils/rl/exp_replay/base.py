import random
from collections import deque


class BaseMemory:
    """
    基于 deque 双端队列用于存储强化学习轨迹
    存储 numpy 数组
    """
    def __init__(self, capacity=2000):
        self.experience = deque(maxlen=capacity)
        # self.now_experience = None    # 强制包含最新transition，可能会引入样本相关性偏差。

    def add(self, state, action, reward, next_state, done):
        self.experience.append((state, action, reward, next_state, done))
        # self.now_experience = (_state, _action, _reward, _next_state, _done)

    def sample(self, _batch_size, order=False):
        if order:
            transitions = list(self.experience)
        else:
            transitions = random.sample(self.experience, _batch_size)
        # transitions.append(self.now_experience)
        return zip(*transitions)

    def clear(self):
        self.experience.clear()

    def __len__(self):
        return len(self.experience)
