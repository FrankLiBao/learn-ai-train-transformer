"""
位置编码 (Positional Encoding)

为序列注入位置信息，使用正弦/余弦函数生成固定编码：
- 偶数维度: sin(pos / 10000^(2i/d_model))
- 奇数维度: cos(pos / 10000^(2i/d_model))

特点：
- 每个位置有唯一的编码
- 相对位置可以通过线性变换表示
- 可以外推到训练时未见过的序列长度
- 不需要学习，计算固定值

输出形状: [seq_len, d_model]，与 token embedding 相加
"""

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    正弦位置编码

    为什么需要位置编码?
    - Transformer 的自注意力机制是位置无关的 (permutation invariant)
    - 即打乱输入顺序，输出也只是相应打乱，不会改变
    - 因此需要显式注入位置信息，让模型知道每个 token 在序列中的位置

    为什么使用正弦/余弦函数?
    - 可以表示相对位置: PE(pos+k) 可以表示为 PE(pos) 的线性函数
    - 值域有界: 始终在 [-1, 1] 范围内，不会随位置增大
    - 可外推: 能处理比训练时更长的序列
    - 不同频率捕获不同尺度的位置关系

    编码可视化 (d_model=8 为例):
    pos=0: [sin(0), cos(0), sin(0), cos(0), ...]  = [0, 1, 0, 1, ...]
    pos=1: [sin(1), cos(1), sin(1/λ), cos(1/λ), ...]  (λ 随维度增大而增大)
    pos=2: [sin(2), cos(2), sin(2/λ), cos(2/λ), ...]
    ...
    低维度变化快 (捕获局部位置)，高维度变化慢 (捕获全局位置)
    """

    def __init__(self, d_model, max_len, device):
        """
        初始化位置编码

        参数:
            d_model: 模型维度，与词嵌入维度相同 (如 512)
            max_len: 支持的最大序列长度 (如 512)
            device: 计算设备 (cpu/cuda)

        位置编码在初始化时一次性计算完成，之后作为常量使用
        """
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵 [max_len, d_model]
        # 预先计算所有位置的编码，推理时直接查表
        self.encoding = torch.zeros(max_len, d_model, device=device)
        # 位置编码是固定的，不参与梯度计算
        self.encoding.requires_grad = False

        # 生成位置索引: [0, 1, 2, ..., max_len-1]
        pos = torch.arange(0, max_len, device=device)
        # 扩展维度: [max_len] -> [max_len, 1]
        # 这样可以与维度索引进行广播运算
        pos = pos.float().unsqueeze(dim=1)

        # 生成维度索引: [0, 2, 4, ..., d_model-2]
        # 公式中的 2i，表示偶数维度的索引
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 计算位置编码
        # 公式: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        #       PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        #
        # 10000^(2i/d_model) 是波长，随着 i 增大而增大
        # - 当 i=0 时，波长=1，变化最快
        # - 当 i=d_model/2 时，波长=10000，变化最慢
        #
        # 广播机制:
        # pos: [max_len, 1], _2i: [d_model/2]
        # pos / (10000 ** (_2i / d_model)): [max_len, d_model/2]

        # 偶数维度使用 sin
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        # 奇数维度使用 cos
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        """
        获取位置编码

        参数:
            x: 输入的 token 索引，形状 [batch_size, seq_len]

        返回:
            位置编码，形状 [seq_len, d_model]
            通过广播机制与 [batch_size, seq_len, d_model] 的词嵌入相加

        注意: 返回的编码会自动广播到 batch 维度
        例如: tok_emb [128, 30, 512] + pos_enc [30, 512] -> [128, 30, 512]
        """
        batch_size, seq_len = x.size()

        # 根据实际序列长度截取位置编码
        # 只取前 seq_len 个位置的编码
        return self.encoding[:seq_len, :]
