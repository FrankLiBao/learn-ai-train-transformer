"""
编码器层 (Encoder Layer)

Transformer 编码器的单层结构：
1. Multi-Head Self-Attention (多头自注意力)
2. Add & Norm (残差连接 + 层归一化)
3. Position-wise Feed-Forward Network (前馈网络)
4. Add & Norm (残差连接 + 层归一化)

输入输出形状保持一致: [batch_size, seq_len, d_model]
"""

from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm1 = LayerNorm(d_model=d_model)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)

    def forward(self, x, s_mask):
        # 1. compute self attention
        # print("encoder layer x: ", x.shape)
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=s_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
