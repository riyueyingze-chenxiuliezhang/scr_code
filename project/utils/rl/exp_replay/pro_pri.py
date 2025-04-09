"""
# @Time: 2025/4/7 18:04
# @File: pro_pri.py

基于比例的优先级（Proportional Prioritization）
"""
import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.empty(capacity, dtype=object)
        self.data_pointer = 0
        self.data_size = 0
        self.max_priority = 1.0     # 初始最大优先级

    def add(self, data, priority):
        self.data[self.data_pointer] = data
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.data_size < self.capacity:
            self.data_size += 1
        if priority > self.max_priority:
            self.max_priority = priority

    def update(self, tree_idx, priority):
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # 向上更新父节点
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def get_leaf(self, value):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def size(self):
        return self.data_size

    @property
    def total_priority(self):
        return self.tree[0]


class ProPriReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        priority = self.tree.max_priority ** self.alpha
        self.tree.add((state, action, reward, next_state, done), priority)

    def sample(self, batch_size):
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(s)
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        probabilities = np.array(priorities) / self.tree.total_priority
        weights = np.power(self.tree.size * probabilities, -self.beta)
        weights /= weights.max()

        self.beta = np.min([1.0, self.beta + self.beta_increment])
        return zip(*batch), indices, weights

    def update_priorities(self, indices, td_errors):
        priorities = np.abs(td_errors) + self.epsilon
        for idx, priority in zip(indices, priorities):
            p_alpha = priority ** self.alpha
            self.tree.update(idx, p_alpha)

    def __len__(self):
        return self.tree.size
