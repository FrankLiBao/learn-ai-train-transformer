"""
Transformer 前向传播演示 (Forward Pass Demo)

逐步演示 Transformer 模型各组件的前向传播过程：
- Mask 生成：pad mask 和 no-peak mask
- Encoder 编码过程：Embedding -> Multi-Head Attention -> FFN
- Decoder 解码过程：Self-Attention -> Cross-Attention -> FFN
- 各层的张量形状变化
- 损失计算示例

用于学习和调试 Transformer 的内部计算流程
"""

import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time

model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

# print(f'The model has {count_parameters(model):,} trainable parameters')
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)
model.apply(initialize_weights)

src = torch.load('data/tensor_src.pt')
trg = torch.load('data/tensor_trg.pt')
print("load src shape", src.shape)
print("load trg shape", trg.shape)

# mask 处理
# Transformer.
src_mask = model.make_pad_mask(src, src, src_pad_idx, src_pad_idx)
src_trg_mask = model.make_pad_mask(trg, src, trg_pad_idx, src_pad_idx)
trg_mask = model.make_pad_mask(trg, trg, trg_pad_idx, trg_pad_idx) * \
            model.make_no_peak_mask(trg, trg)

# # transformer 编码层和解码层计算
# enc_src = model.encoder(src, src_mask)
# output = model.decoder(trg, enc_src, trg_mask, src_trg_mask)

# encoder 编码层计算
emb_src = model.encoder.emb(src)
for layer in model.encoder.layers:
    encoder_src = layer(emb_src, src_mask)

# encoder layer blocks
for layer in model.encoder.layers:
    # encoder_src = layer(emb_src, src_mask)
    _emb_src = emb_src
    
    # mutiattention
    x = layer.attention(q=emb_src, k=emb_src, v=emb_src, mask=src_mask)

    # 2. add and norm
    x = layer.dropout1(x)
    x = layer.norm1(x + _emb_src)

    # 3. positionwise feed forward network
    _x = x
    x = layer.ffn(x)

    # 4. add and norm
    x = layer.dropout2(x)
    x = layer.norm2(x + _x)
    
    break

# encode multi-attention
multi_head_attention = model.encoder.layers[0].attention
x_attention_out = multi_head_attention(q=emb_src, k=emb_src, v=emb_src, mask=src_mask)

# self attention
q = k = v = emb_src

# 1. dot product with weight matrices
q = multi_head_attention.w_q(q)
k = multi_head_attention.w_k(k)
v = multi_head_attention.w_v(v)

# 2. split tensor by number of heads
_q = q
# do split
batch_size, length, d_model = _q.size()
d_tensor = d_model // multi_head_attention.n_head
_q_split = tensor.view(batch_size, length, multi_head_attention.n_head, d_tensor).transpose(1, 2)

print("before head to multi_head: q,k,v shape:", q.shape)
q, k, v = multi_head_attention.split(q), multi_head_attention.split(k), multi_head_attention.split(v)
print("after head to multi_head: q,k,v shape:", q.shape)

# 3. do scale dot product to compute similarity
# 第一个为多头attention
# 第二个为单头attention（scale and dot attention）
_q_single = q
_k_single = k
_v_single = v
out, attention = multi_head_attention.attention(q, k, v, mask=mask)
print("after sclae dot prodtuct out shape:", out.shape)

# 4. concat and pass to linear layer
_out = out 
# do concat 
batch_size, head, length, d_tensor = _out.size()
d_model = head * d_tensor
_out_concat = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)

out = multi_head_attention.concat(out)
print("after concat out shape:", out.shape)
out = multi_head_attention.w_concat(out)
print("after w_concat out shape:", out.shape)




# attention
# class ScaleDotProductAttention(nn.Module):

# input is 4 dimension tensor
# [batch_size, head, length, d_tensor]
k = _k_single
q = _q_single
v = _v_single
batch_size, head, length, d_tensor = k.size()

# 1. dot product Query with Key^T to compute similarity
k_t = k.transpose(2, 3)  # transpose

print("======scale dot product======")
print("q:", q.shape)
print("k_t:", k_t.shape)
score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

# 2. apply masking (opt)
if mask is not None:
    score = score.masked_fill(mask == 0, -10000)

# 3. pass them softmax to make [0, 1] range
score = self.softmax(score)

# 4. multiply with Value
v = score @ v
print("v:", v.shape)


## layer normalization
# self.gamma = nn.Parameter(torch.ones(d_model))
# self.beta = nn.Parameter(torch.zeros(d_model))
# self.eps = eps

norm = model.encoder.layers[0].norm1
x = emb_src
print("==============LayerNorm===========")
print("LayerNorm gamma: ", norm.gamma.shape)
print("LayerNorm beta: ", norm.beta.shape)
print("LayerNorm eps: ", norm.eps)

mean = x.mean(-1, keepdim=True)
print("LayerNorm mean: ", mean.shape)

var = x.var(-1, unbiased=False, keepdim=True)
print("LayerNorm var: ", var.shape)
# '-1' means last dimension. 

out = (x - mean) / torch.sqrt(var + norm.eps)
print("LayerNorm norm out: ", out.shape)

out = norm.gamma * out + norm.beta
print("LayerNorm norm out offset: ", out.shape)


# positional forward 
ffn = model.encoder.layers[0].ffn
_x = emb_src
_x = ffn.linear1(_x)
_x = ffn.relu(_x)
_x = ffn.dropout(_x)
_x = ffn.linear2(_x)

# decoder layer
emb_trg = model.emb(trg)
for layer in model.decoder.layers:
    trg = layer(emb_trg, enc_src, trg_mask, src_mask)
# pass to LM head
output_decode = model.decoder.linear(trg)

# decode layer
layer = model.decoder.layers[0]
dec = emb_trg
enc = enc_src
_x = dec

x = layer.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
x = layer.dropout1(x)
x = layer.norm1(x + _x)

if enc is not None:
    # 3. compute encoder - decoder attention
    _x = x
    # 多头注意力机制
    x = layer.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
    # 4. add and norm
    x = layer.dropout2(x)
    x = layer.norm2(x + _x)

# 5. positionwise feed forward network
_x = x
x = layer.ffn(x)

# 6. add and norm
x = layer.dropout3(x)
x = layer.norm3(x + _x)


## 损失计算
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
output = model(src, trg[:, :-1])
output_reshape = output.contiguous().view(-1, output.shape[-1])
trg = trg[:, 1:].contiguous().view(-1)
loss = criterion(output_reshape, trg)
loss.backward()

epoch = 10
for i in range(epoch):
    output = model(src, trg[:, :-1])
    output_reshape = output.contiguous().view(-1, output.shape[-1])
    trg = trg[:, 1:].contiguous().view(-1)
    loss = criterion(output_reshape, trg)
    loss.backward()
    print(loss.item)