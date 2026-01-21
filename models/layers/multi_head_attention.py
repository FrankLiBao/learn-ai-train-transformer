"""
多头注意力机制 (Multi-Head Attention)

将注意力机制并行化到多个"头"上：
1. 线性变换: Q, K, V 分别通过 W_q, W_k, W_v 投影
2. 分割多头: 将 d_model 维度分割成 n_head 个 d_k 维度
3. 并行注意力: 每个头独立计算 Scaled Dot-Product Attention
4. 拼接输出: 将多头输出拼接并通过 W_o 投影

公式: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
其中: head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
"""

from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制

    为什么需要多头?
    - 单头注意力只能学习一种注意力模式
    - 多头允许模型同时关注不同位置的不同表示子空间
    - 例如: 一个头关注语法关系，另一个头关注语义关系

    计算流程:
    输入 [batch, seq_len, d_model]
      ↓ 线性投影 (W_q, W_k, W_v)
    Q, K, V [batch, seq_len, d_model]
      ↓ 分割成多头
    Q, K, V [batch, n_head, seq_len, d_k]  其中 d_k = d_model / n_head
      ↓ 每个头独立计算注意力
    output [batch, n_head, seq_len, d_k]
      ↓ 拼接多头
    output [batch, seq_len, d_model]
      ↓ 输出投影 (W_o)
    output [batch, seq_len, d_model]
    """

    def __init__(self, d_model, n_head):
        """
        初始化多头注意力层

        参数:
            d_model: 模型的隐藏维度 (如 512)
            n_head: 注意力头的数量 (如 8)

        注意: d_model 必须能被 n_head 整除
              每个头的维度 d_k = d_model / n_head (如 512/8=64)
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        # Q, K, V 的线性投影矩阵，将输入映射到查询/键/值空间
        self.w_q = nn.Linear(d_model, d_model)  # W^Q: 查询投影
        self.w_k = nn.Linear(d_model, d_model)  # W^K: 键投影
        self.w_v = nn.Linear(d_model, d_model)  # W^V: 值投影
        # 缩放点积注意力模块
        self.attention = ScaleDotProductAttention()
        # 输出投影矩阵 W^O，将拼接后的多头输出映射回 d_model 维度
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        前向传播

        参数:
            q: 查询输入，形状 [batch_size, seq_len, d_model]
            k: 键输入，形状 [batch_size, seq_len, d_model]
            v: 值输入，形状 [batch_size, seq_len, d_model]
            mask: 注意力掩码 (可选)

        返回:
            out: 多头注意力输出，形状 [batch_size, seq_len, d_model]

        注意: 在自注意力中 q=k=v，在交叉注意力中 q 来自解码器，k=v 来自编码器
        """
        # 步骤1: 通过线性层对 Q, K, V 进行投影
        # 每个线性层学习不同的投影，使 Q, K, V 进入各自的表示空间
        # 形状保持: [batch, seq_len, d_model]
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 步骤2: 将张量分割成多个头
        # [batch, seq_len, d_model] -> [batch, n_head, seq_len, d_k]
        # 这样每个头可以独立计算注意力
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 步骤3: 对每个头并行计算缩放点积注意力
        # 输入输出形状: [batch, n_head, seq_len, d_k]
        # attention 是注意力权重矩阵，可用于可视化
        out, attention = self.attention(q, k, v, mask=mask)

        # 步骤4: 拼接所有头的输出，并通过输出投影层
        # [batch, n_head, seq_len, d_k] -> [batch, seq_len, d_model]
        out = self.concat(out)
        # 输出投影，允许不同头的信息融合
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        将张量分割成多个注意力头

        将最后一个维度 d_model 分割成 n_head 个 d_k
        然后调整维度顺序，使 head 维度在 seq_len 之前

        参数:
            tensor: 输入张量，形状 [batch_size, seq_len, d_model]

        返回:
            tensor: 分割后的张量，形状 [batch_size, n_head, seq_len, d_k]

        示例 (d_model=512, n_head=8, d_k=64):
            [32, 100, 512] -> view -> [32, 100, 8, 64] -> transpose -> [32, 8, 100, 64]
        """
        batch_size, length, d_model = tensor.size()

        # 计算每个头的维度: d_k = d_model / n_head
        d_tensor = d_model // self.n_head

        # view: [batch, length, d_model] -> [batch, length, n_head, d_k]
        # transpose(1,2): 交换 length 和 n_head 维度
        # 结果: [batch, n_head, length, d_k]
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # 这个操作类似于分组卷积的思想，将通道分成多个组独立处理

        return tensor

    def concat(self, tensor):
        """
        拼接多个注意力头的输出 (split 的逆操作)

        将所有头的输出拼接回 d_model 维度

        参数:
            tensor: 多头输出，形状 [batch_size, n_head, seq_len, d_k]

        返回:
            tensor: 拼接后的张量，形状 [batch_size, seq_len, d_model]

        示例 (d_model=512, n_head=8, d_k=64):
            [32, 8, 100, 64] -> transpose -> [32, 100, 8, 64] -> view -> [32, 100, 512]
        """
        batch_size, head, length, d_tensor = tensor.size()
        # 恢复 d_model = n_head * d_k
        d_model = head * d_tensor

        # transpose(1,2): [batch, n_head, length, d_k] -> [batch, length, n_head, d_k]
        # contiguous(): 使内存连续，view 操作需要连续内存
        # view: [batch, length, n_head, d_k] -> [batch, length, d_model]
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
