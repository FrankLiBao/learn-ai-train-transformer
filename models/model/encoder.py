"""
Transformer 编码器 (Encoder)

将输入序列编码为上下文表示：
- TransformerEmbedding: 词嵌入 + 位置编码
- N 层 EncoderLayer 堆叠
- 每层包含: Multi-Head Self-Attention + Feed-Forward Network
- 输出形状: [batch_size, seq_len, d_model]
"""

from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, s_mask):

        # print("before emding x", x.shape)
        x = self.emb(x)
        # print("after emding x", x.shape)

        for layer in self.layers:
            x = layer(x, s_mask)

        return x
