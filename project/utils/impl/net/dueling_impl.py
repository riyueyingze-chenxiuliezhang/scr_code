"""
# @Time: 2025/4/7 14:53
# @File: dueling_impl.py
"""
from typing import List

from torch import nn


class DuelingNet(nn.Module):
    def __init__(self, _input_size, _output_size, _hidden_layer: List[int] = None):
        super().__init__()

        # 默认隐藏层设置
        _hidden_layer = _hidden_layer if _hidden_layer else [64]
        assert isinstance(_hidden_layer, List), "隐藏层必须是一个列表"
        assert all(isinstance(n, int) and n > 0 for n in _hidden_layer), "隐藏层维度需为正整数"

        # 构建共享特征提取层：从输入到最后一个隐藏层
        shared_layer_dims = [_input_size, *_hidden_layer]
        self.shared_layers = nn.ModuleList()
        for in_dim, out_dim in zip(shared_layer_dims[:-1], shared_layer_dims[1:]):
            self.shared_layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU()
            ))

        # 两个分支头：状态价值和优势
        last_hidden = _hidden_layer[-1]
        self.value_layer = nn.Linear(last_hidden, 1)
        self.advantage_layer = nn.Linear(last_hidden, _output_size)

    def forward(self, x):
        # 共享特征层
        for layer in self.shared_layers:
            x = layer(x)
        # 计算状态价值 V(s) 和优势 A(s,a)
        value = self.value_layer(x)                    # shape: (batch, 1)
        advantage = self.advantage_layer(x)            # shape: (batch, _output_size)
        # 组合成 Q(s,a)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
