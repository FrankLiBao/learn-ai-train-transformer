"""
分词器 (Tokenizer)

使用 spaCy 对英语和德语文本进行分词：
- tokenize_en: 英语分词
- tokenize_de: 德语分词

依赖 spaCy 模型:
- en_core_web_sm (英语)
- de_core_news_sm (德语)

安装: python -m spacy download en_core_web_sm
      python -m spacy download de_core_news_sm
"""

import spacy


class Tokenizer:

    def __init__(self):
        # python -m spacy download de_core_news_sm
        self.spacy_de = spacy.load('de_core_news_sm')
        # python -m spacy download en_core_web_sm
        self.spacy_en = spacy.load('en_core_web_sm')
        # using conda conda install -c conda-forge spacy-model-de_core_web_sm

    def tokenize_de(self, text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]
        # example
        # doc = nlp('This is an example sentence.')
        # tokens = [token.text for token in doc]
        # print(tokens)
        # ['This', 'is', 'an', 'example', 'sentence', '.']