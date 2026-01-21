"""
位置前馈网络 (Position-wise Feed-Forward Network)

对序列中每个位置独立应用相同的全连接网络：
1. 第一层线性变换: d_model -> hidden (扩展维度)
2. ReLU 激活函数
3. Dropout 正则化
4. 第二层线性变换: hidden -> d_model (恢复维度)

公式: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

"Position-wise" 表示对每个位置独立计算，
即沿序列维度共享参数，沿特征维度独立计算
"""

from torch import nn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        self.linear2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
