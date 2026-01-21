"""
词嵌入 (Token Embedding)

将离散的词索引映射为稠密的向量表示：
- 输入: token 索引 [batch_size, seq_len]
- 输出: 嵌入向量 [batch_size, seq_len, d_model]
- padding_idx=1 表示 padding token 的嵌入向量始终为零向量

继承自 nn.Embedding，使用可学习的嵌入矩阵
"""

from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
