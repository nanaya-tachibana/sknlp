from __future__ import annotations

try:
    import jieba_fast as jieba
except ImportError:
    import jieba
import tensorflow as tf

from sknlp.vocab import Vocab
from sknlp.layers import BertPreprocessingLayer


class Tokenizer:
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    def tokenize(self, text: str) -> list[int]:
        raise NotImplementedError()


class JiebaTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[int]:
        tokens = jieba.lcut(text, HMM=False)
        return self.vocab.token2idx(tokens)


class CharTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[int]:
        tokens = list(text)
        return self.vocab.token2idx(tokens)


class BertTokenizer(Tokenizer):
    def __init__(self, vocab: Vocab) -> None:
        super().__init__(vocab)
        layer = BertPreprocessingLayer(self.vocab.sorted_tokens)
        layer.build(tf.TensorShape([None]))
        self._tokenizer = layer.tokenizer

    def tokenize(self, text: str) -> list[int]:
        token_ids: tf.RaggedTensor = self._tokenizer.tokenize([text])
        token_ids = token_ids.merge_dims(-2, -1)
        return token_ids.to_list()[0]


def get_tokenizer(name: str, vocab: Vocab) -> Tokenizer:
    if name == "jieba":
        return JiebaTokenizer(vocab)
    elif name == "list" or name == "char":
        return CharTokenizer(vocab)
    elif name is None:
        return BertTokenizer(vocab)
    else:
        raise ValueError("unknown segmenter %s" % name)