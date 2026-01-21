# Transformer 从零实现

基于论文 "Attention Is All You Need" 的 Transformer 模型 PyTorch 实现，用于英德机器翻译任务。

## 模型架构

![Transformer Architecture](image/model.png)

Transformer 模型采用 Encoder-Decoder 架构，完全基于注意力机制，摒弃了传统的 RNN/CNN 结构。

### 整体结构

![Encoder-Decoder](image/enc_dec.jpg)

- **Encoder**: 由 N 个相同的层堆叠而成，每层包含多头自注意力和前馈网络
- **Decoder**: 同样由 N 个相同的层堆叠，每层包含掩码自注意力、编码器-解码器注意力和前馈网络

## 核心组件

### 1. Scaled Dot-Product Attention

![Scaled Dot-Product Attention](image/scale_dot_product_attention.jpg)

注意力机制的核心计算公式：

$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

```python
# 实现位置: models/layers/scale_dot_product_attention.py
score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
score = self.softmax(score)
output = score @ v
```

### 2. Multi-Head Attention

![Multi-Head Attention](image/multi_head_attention.jpg)

多头注意力允许模型同时关注不同位置的不同表示子空间：

$$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$$
$$where\ head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

```python
# 实现位置: models/layers/multi_head_attention.py
q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)  # 线性变换
q, k, v = self.split(q), self.split(k), self.split(v)  # 分割成多头
out, attention = self.attention(q, k, v, mask=mask)  # 计算注意力
out = self.concat(out)  # 拼接
out = self.w_concat(out)  # 输出投影
```

### 3. Positional Encoding

![Positional Encoding](image/positional_encoding.jpg)

由于模型不包含循环结构，需要注入位置信息：

$$PE_{(pos, 2i)} = sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = cos(pos / 10000^{2i/d_{model}})$$

```python
# 实现位置: models/embedding/positional_encoding.py
self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
```

### 4. Layer Normalization

![Layer Norm](image/layer_norm.jpg)

Layer Norm 对每个样本的特征维度进行归一化：

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

```python
# 实现位置: models/layers/layer_norm.py
mean = x.mean(-1, keepdim=True)
var = x.var(-1, unbiased=False, keepdim=True)
out = (x - mean) / torch.sqrt(var + self.eps)
out = self.gamma * out + self.beta
```

### 5. Position-wise Feed-Forward Network

![Feed Forward](image/positionwise_feed_forward.jpg)

$$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$$

```python
# 实现位置: models/layers/position_wise_feed_forward.py
x = self.linear1(x)
x = self.relu(x)
x = self.dropout(x)
x = self.linear2(x)
```

## 项目结构

```
├── conf.py                 # 配置文件（超参数设置）
├── data.py                 # 数据加载与预处理
├── train.py                # 训练脚本
├── models/
│   ├── model/
│   │   ├── transformer.py  # Transformer 主模型
│   │   ├── encoder.py      # 编码器
│   │   └── decoder.py      # 解码器
│   ├── blocks/
│   │   ├── encoder_layer.py    # 编码器层
│   │   └── decoder_layer.py    # 解码器层
│   ├── layers/
│   │   ├── multi_head_attention.py         # 多头注意力
│   │   ├── scale_dot_product_attention.py  # 缩放点积注意力
│   │   ├── position_wise_feed_forward.py   # 前馈网络
│   │   └── layer_norm.py                   # 层归一化
│   └── embedding/
│       ├── token_embeddings.py      # 词嵌入
│       ├── positional_encoding.py   # 位置编码
│       └── transformer_embedding.py # 组合嵌入
├── util/
│   ├── data_loader.py      # 数据加载器
│   ├── tokenizer.py        # 分词器
│   ├── bleu.py             # BLEU 评估指标
│   └── epoch_timer.py      # 计时工具
└── image/                  # 架构图
```

## 模型参数

![Model Size](image/transformer-model-size.jpg)

本项目实现的是 Transformer Base 配置：

| 参数 | 值 | 说明 |
|------|------|------|
| `d_model` | 512 | 模型维度 |
| `n_layers` | 6 | 编码器/解码器层数 |
| `n_heads` | 8 | 注意力头数 |
| `d_k, d_v` | 64 | 每个头的维度 (512/8) |
| `ffn_hidden` | 2048 | 前馈网络隐藏层维度 |
| `drop_prob` | 0.1 | Dropout 概率 |
| `max_len` | 256 | 最大序列长度 |
| `batch_size` | 128 | 批次大小 |

## 使用方法

### 环境要求

| 依赖包 | 版本 | 说明 |
|--------|------|------|
| Python | 3.12 | 运行环境 |
| torch | 2.9.0 | PyTorch 深度学习框架 |
| torchtext | 0.6.0 | 文本数据处理 |
| spacy | 3.8.11 | NLP 分词工具 |
| numpy | 2.0.2 | 数值计算 |
| matplotlib | 3.10.0 | 可视化绘图 |
| nltk | 3.9.1 | 自然语言工具包 |
| jiwer | 3.1.0 | 词错误率计算 |
| rouge | 1.0.1 | ROUGE 评估指标 |

### 安装依赖

```bash
# 安装依赖包
pip install -r requirements.txt

# 下载 spaCy 语言模型
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### 训练模型

```bash
python train.py
```

训练过程会：
- 自动下载 Multi30k 英德翻译数据集
- 使用 BLEU 分数评估翻译质量
- 保存最佳模型到 `saved/` 目录
- 记录训练损失到 `result/` 目录

## 训练配置

主要超参数在 `conf.py` 中配置：

```python
# 优化器参数
init_lr = 1e-5          # 初始学习率
weight_decay = 5e-4     # 权重衰减
adam_eps = 5e-9         # Adam epsilon

# 学习率调度
factor = 0.9            # 学习率衰减因子
patience = 10           # 衰减等待轮数
warmup = 100            # warmup 轮数

# 训练参数
epoch = 100             # 总训练轮数
clip = 1.0              # 梯度裁剪阈值
```

## 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
