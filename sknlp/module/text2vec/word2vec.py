from __future__ import annotations
from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Embedding

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
        """
        基础符号->向量模块.

        Parameters:
        ----------
        vocab: sknlp.vocab.Vocab. 符号表.
        embed_size: int > 0. 向量维度.
        name: str. 模块名.
        embeddings_initializer: Initializer for the `embeddings` matrix.
        embeddings_regularizer: Regularizer function applied to
        the `embeddings` matrix.
        embeddings_constraint: Constraint function applied to
        the `embeddings` matrix.
        input_length: Length of input sequences, when it is constant.
        This argument is required if you are going to connect
        `Flatten` then `Dense` layers upstream
        (without it, the shape of the dense outputs cannot be computed).

        Inputs
        ----------
        2D tensor with shape: `(batch_size, input_length)`.

        Outputs
        ----------
        3D tensor with shape: `(batch_size, input_length, embed_size)`.
        """
        super().__init__(
            vocab,
            segmenter=segmenter,
            max_sequence_length=max_sequence_length,
            sequence_length=sequence_length,
            embedding_size=embedding_size,
            name=name,
            **kwargs,
        )
        self.embedding = Embedding(
            len(vocab),
            embedding_size,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=True,
            name="embeddings",
        )
        self._model = tf.keras.Sequential(self.embedding, name=name)

    def compute_mask(
        self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        return self._model.get_layer("embeddings").compute_mask(inputs, mask=mask)