"""
配置文件 (Configuration)

定义 Transformer 模型的所有超参数配置，包括：
- 模型结构参数：d_model, n_layers, n_heads, ffn_hidden 等
- 训练参数：batch_size, epoch, learning rate 等
- 优化器参数：Adam 优化器相关设置
- 学习率调度参数：warmup_steps, lr_factor (原论文调度策略)
"""

import torch

# ====================
# 设备配置
# ====================
# 自动选择 GPU (cuda) 或 CPU，优先使用 GPU 加速训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")  # macOS Apple Silicon

# ====================
# 模型结构参数
# ====================
batch_size = 128    # 批次大小：每次训练使用的样本数量
max_len = 256       # 最大序列长度：输入/输出序列的最大 token 数
d_model = 512       # 模型维度：词嵌入和各层输出的向量维度
n_layers = 6        # 层数：Encoder 和 Decoder 各堆叠 6 层
n_heads = 8         # 注意力头数：Multi-Head Attention 的并行头数 (d_k = d_model / n_heads = 64)
ffn_hidden = 2048   # 前馈网络隐藏层维度：FFN 中间层的大小 (通常是 d_model 的 4 倍)
drop_prob = 0.1     # Dropout 概率：随机丢弃 10% 的神经元，防止过拟合

# ====================
# 优化器参数
# ====================
adam_eps = 5e-9     # Adam epsilon：防止除零的小常数，增加数值稳定性
warmup_steps = 100  # warmup 步数：学习率线性增长的步数
lr_factor = 1.0     # 学习率缩放因子：可调节学习率峰值
epoch = 100         # 总训练轮数：完整遍历训练集的次数
clip = 1.0          # 梯度裁剪阈值：防止梯度爆炸，将梯度范数限制在 1.0 以内
weight_decay = 5e-4 # 权重衰减 (L2 正则化)：防止过拟合，惩罚过大的权重
inf = float('inf')  # 正无穷：用于初始化最佳损失值
