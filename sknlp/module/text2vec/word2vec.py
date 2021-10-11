from __future__ import annotations
from typing import Optional


import tensorflow as tf
from tensorflow.keras.layers import Embedding
import numpy as np

from sknlp.vocab import Vocab
from sknlp.typing import WeightRegularizer, WeightInitializer, WeightConstraint

from .text2vec import Text2vec


class Word2vec(Text2vec):
    def __init__(
        self,
        vocab: Vocab,
        embedding_size: int,
        segmenter: str = "jieba",
        max_sequence_length: Optional[int] = None,
        sequence_length: Optional[int] = None,
        embeddings_initializer: WeightInitializer = "uniform",
        embeddings_regularizer: Optional[WeightRegularizer] = None,
        embeddings_constraint: Optional[WeightConstraint] = None,
        name: str = "word2vec",
        **kwargs,
    ) -> None:
        super().__init__(
            vocab,
            segmenter=segmenter,
            max_sequence_length=max_sequence_length,
            sequence_length=sequence_length,
            embedding_size=embedding_size,
            algorithm="word2vec",
            name=name,
            **kwargs,
        )
        self.pretrain_layer = Embedding(
            len(vocab),
            embedding_size,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=True,
            name=self.name,
        )
        self.inputs = tf.keras.Input(shape=(None,), dtype=tf.int64, name="token_ids")

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.pretrain_layer(inputs)

    def build_output_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.pretrain_layer(inputs)

    def to_word2vec_format(self, filename: str) -> None:
        with open(filename, "w") as f:
            for token, vec in zip(
                self.vocab.sorted_tokens, self._model.weights[0].numpy()
            ):
                f.write(" ".join([token, " ".join(map(str, vec))]))
                f.write("\n")

    @classmethod
    def from_word2vec_format(
        cls, filename: str, segmenter: str = "jieba"
    ) -> "Word2vec":
        pad = "<pad>"
        unk = "<unk>"
        bos = "<s>"
        eos = "</s>"
        num_special_tokens = 0
        vocab_size = 0
        embedding_size = 0
        has_header = False
        has_unk = False
        with open(filename) as f:
            for line in f:
                line = line.strip("\n")
                if line == "":
                    continue
                if embedding_size == 0 and not line.startswith(" "):
                    if len(line.split()) == 2:
                        embedding_size = int(line.split(" ")[1])
                        has_header = True
                        continue
                    else:
                        embedding_size = len(line.split(" ")) - 1
                if (
                    line.startswith(pad)
                    or line.startswith(unk)
                    or line.startswith(bos)
                    or line.startswith(eos)
                ):
                    num_special_tokens += 1
                vocab_size += 1

        num_adding_tokens = 4 - num_special_tokens
        weights = np.zeros((vocab_size + num_adding_tokens, embedding_size))
        weights[2:4, :] = np.random.uniform(-0.1, 0.1, size=(2, embedding_size))

        tokens = [pad, unk, bos, eos]
        with open(filename) as f:
            for i, line in enumerate(f):
                line = line.strip("\n")
                if line == "":
                    continue
                if has_header:
                    i -= 1
                if i < 0:
                    continue

                cells = line.split(" ")
                if cells[0] == "":
                    del cells[0]
                    cells[0] = " "
                token = cells[0]
                idx = i + num_adding_tokens + num_special_tokens
                if token == pad:
                    idx = 0
                    num_special_tokens -= 1
                elif token == unk:
                    idx = 1
                    has_unk = True
                    num_special_tokens -= 1
                elif token == bos:
                    idx = 2
                    num_special_tokens -= 1
                elif token == eos:
                    idx = 3
                    num_special_tokens -= 1
                else:
                    tokens.append(token)
                vec = list(map(float, cells[1:]))
                weights[idx, :] = vec
        vocab = Vocab(
            tokens,
            pad_token=pad,
            unk_token=unk,
            bos_token=bos,
            eos_token=eos,
        )
        if not has_unk:
            weights[1, :] = weights[4:, :].mean(axis=0)
        word2vec = cls(vocab, embedding_size, segmenter=segmenter)
        word2vec.build()
        word2vec._model.set_weights([weights])
        return word2vec