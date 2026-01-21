"""
数据加载器 (Data Loader) - 使用 HuggingFace datasets

从 HuggingFace 加载 Multi30k 英德翻译数据集，
保持与 torchtext 相同的接口，确保 train.py 无需修改。
"""

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import Counter


class Vocab:
    """词汇表，兼容 torchtext.vocab 接口"""
    def __init__(self, tokens, specials):
        self.stoi = {tok: i for i, tok in enumerate(specials)}
        self.itos = list(specials)
        for tok, freq in tokens.items():
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def __len__(self):
        return len(self.itos)


class Batch:
    """批次对象，保持 batch.src / batch.trg 接口，使 train.py 无需修改"""
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg


class DataLoader:
    source = None
    target = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.init_token = init_token
        self.eos_token = eos_token
        self.src_lang, self.trg_lang = ('en', 'de') if ext == ('.en', '.de') else ('de', 'en')
        self.src_tokenize = tokenize_en if self.src_lang == 'en' else tokenize_de
        self.trg_tokenize = tokenize_de if self.trg_lang == 'de' else tokenize_en
        print('dataset initializing start')

    def make_dataset(self, cache_dir='./data'):
        ds = load_dataset('bentrevett/multi30k', cache_dir=cache_dir)
        return ds['train'], ds['validation'], ds['test']

    def build_vocab(self, train_data, min_freq=2):
        specials = ['<pad>', '<sos>', '<eos>', '<unk>']

        # 统计词频
        src_counter, trg_counter = Counter(), Counter()
        for item in train_data:
            src_counter.update(self.src_tokenize(item[self.src_lang].lower()))
            trg_counter.update(self.trg_tokenize(item[self.trg_lang].lower()))

        # 过滤低频词
        src_tokens = {k: v for k, v in src_counter.items() if v >= min_freq}
        trg_tokens = {k: v for k, v in trg_counter.items() if v >= min_freq}

        # 创建词汇表
        self.source = type('Field', (), {'vocab': Vocab(src_tokens, specials)})()
        self.target = type('Field', (), {'vocab': Vocab(trg_tokens, specials)})()
        print(f'Source vocab: {len(self.source.vocab)}, Target vocab: {len(self.target.vocab)}')

    def make_iter(self, train, valid, test, batch_size, device):
        self.device = device

        def collate(batch):
            src_list, trg_list = [], []
            pad = self.source.vocab.stoi['<pad>']
            sos = self.target.vocab.stoi['<sos>']
            eos = self.target.vocab.stoi['<eos>']
            unk = self.source.vocab.stoi['<unk>']

            for item in batch:
                src_ids = [self.source.vocab.stoi.get(t, unk) for t in self.src_tokenize(item[self.src_lang].lower())]
                trg_ids = [self.target.vocab.stoi.get(t, unk) for t in self.trg_tokenize(item[self.trg_lang].lower())]
                src_list.append(torch.tensor(src_ids))
                trg_list.append(torch.tensor([sos] + trg_ids + [eos]))

            src = pad_sequence(src_list, batch_first=True, padding_value=pad)
            trg = pad_sequence(trg_list, batch_first=True, padding_value=pad)
            return Batch(src.to(device), trg.to(device))

        train_iter = TorchDataLoader(list(train), batch_size=batch_size, shuffle=True, collate_fn=collate)
        valid_iter = TorchDataLoader(list(valid), batch_size=batch_size, shuffle=False, collate_fn=collate)
        test_iter = TorchDataLoader(list(test), batch_size=batch_size, shuffle=False, collate_fn=collate)
        print('dataset initializing done')
        return train_iter, valid_iter, test_iter
