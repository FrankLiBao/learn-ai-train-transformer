"""
层归一化 (Layer Normalization)

对每个样本的特征维度进行归一化：
1. 计算均值和方差 (沿最后一个维度)
2. 标准化: (x - mean) / sqrt(var + eps)
3. 仿射变换: gamma * x + beta (可学习参数)

与 Batch Norm 不同，Layer Norm 对单个样本归一化，
适用于序列模型，不依赖 batch size

公式: y = gamma * (x - mean) / sqrt(var + eps) + beta
"""

import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
    
