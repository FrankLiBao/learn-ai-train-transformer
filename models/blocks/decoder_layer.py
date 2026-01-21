"""
解码器层 (Decoder Layer)

Transformer 解码器的单层结构：
1. Masked Multi-Head Self-Attention (掩码自注意力，防止看到未来信息)
2. Add & Norm (残差连接 + 层归一化)
3. Multi-Head Cross-Attention (编码器-解码器注意力)
4. Add & Norm (残差连接 + 层归一化)
5. Position-wise Feed-Forward Network (前馈网络)
6. Add & Norm (残差连接 + 层归一化)

输入输出形状保持一致: [batch_size, seq_len, d_model]
"""

from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm1 = LayerNorm(d_model=d_model)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)

    def forward(self, dec, enc, t_mask, s_mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=t_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=s_mask)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
