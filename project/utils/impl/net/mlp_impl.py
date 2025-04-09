"""
# @Time: 2025/4/7 14:52
# @File: mpl_impl.py
"""
from typing import List

from torch import nn


class MLPNet(nn.Module):
    def __init__(self, _input_size, _output_size, _hidden_layer: List[int] = None):
        super().__init__()

        # 检查参数是否合法
        _hidden_layer = _hidden_layer if _hidden_layer else [64]
        assert isinstance(_hidden_layer, List), "隐藏层必须是一个列表"
        assert all(isinstance(n, int) and n > 0 for n in _hidden_layer), "隐藏层维度需为正整数"

        layer_dims = [_input_size, *_hidden_layer]
        self.layers = nn.ModuleList()

        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU()
            ))
        self.layers.append(nn.Linear(_hidden_layer[-1], _output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)
