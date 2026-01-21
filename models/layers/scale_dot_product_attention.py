"""
缩放点积注意力 (Scaled Dot-Product Attention)

注意力机制的核心计算单元：
1. 计算相似度: Q 与 K^T 的点积
2. 缩放: 除以 sqrt(d_k) 防止梯度消失
3. 掩码 (可选): 对无效位置填充极小值
4. Softmax: 归一化得到注意力权重
5. 加权求和: 注意力权重与 V 相乘

公式: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V

输入形状: [batch_size, n_head, seq_len, d_k]
输出形状: [batch_size, n_head, seq_len, d_k]
"""

import math

from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    缩放点积注意力机制

    Query (查询): 当前位置需要关注的内容表示 (通常来自解码器)
    Key (键):    用于计算与 Query 相似度的内容表示 (通常来自编码器)
    Value (值):  实际被聚合的内容表示，与 Key 一一对应 (通常来自编码器)

    注意力计算流程:
    1. Q 和 K^T 做点积，得到原始注意力分数
    2. 除以 sqrt(d_k) 进行缩放，防止点积值过大导致 softmax 梯度消失
    3. 应用掩码 (可选)，将无效位置设为极小值
    4. softmax 归一化，得到注意力权重 (和为1)
    5. 用注意力权重对 V 加权求和，得到输出
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        # 在最后一个维度上做 softmax，即对每个查询位置的所有键位置做归一化
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        """
        前向传播

        参数:
            q: Query 张量，形状 [batch_size, n_head, seq_len, d_k]
            k: Key 张量，形状 [batch_size, n_head, seq_len, d_k]
            v: Value 张量，形状 [batch_size, n_head, seq_len, d_k]
            mask: 掩码张量，0 表示需要被屏蔽的位置 (可选)
            e: 数值稳定性的小常数 (未使用)

        返回:
            output: 注意力加权后的输出，形状 [batch_size, n_head, seq_len, d_k]
            score: 注意力权重矩阵，形状 [batch_size, n_head, seq_len, seq_len]
        """
        # 输入是4维张量: [批次大小, 注意力头数, 序列长度, 每个头的维度]
        batch_size, head, length, d_tensor = k.size()

        # 步骤1: 计算 Q 和 K^T 的点积，得到注意力分数
        # k 的形状: [batch, head, seq_len, d_k]
        # k_t 转置后: [batch, head, d_k, seq_len]
        k_t = k.transpose(2, 3)

        # Q @ K^T 的形状: [batch, head, seq_len, seq_len]
        # score[i,j] 表示第 i 个位置对第 j 个位置的注意力分数
        # 除以 sqrt(d_k) 是为了防止点积值过大，使 softmax 梯度更稳定
        score = (q @ k_t) / math.sqrt(d_tensor)

        # 步骤2: 应用掩码 (可选)
        # 将 mask 为 0 的位置填充为 -10000 (近似负无穷)
        # 这样经过 softmax 后这些位置的注意力权重接近 0
        # 常用于: padding mask (屏蔽填充位置) 和 look-ahead mask (解码器中防止看到未来信息)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 步骤3: softmax 归一化
        # 将分数转换为概率分布 (每行和为1)
        # 分数高的位置获得更大的注意力权重
        score = self.softmax(score)

        # 步骤4: 用注意力权重对 Value 进行加权求和
        # score: [batch, head, seq_len, seq_len]
        # v: [batch, head, seq_len, d_k]
        # 输出: [batch, head, seq_len, d_k]
        # 每个位置的输出是所有 Value 向量的加权平均
        v = score @ v

        return v, score
