from __future__ import annotations
from typing import Optional, Sequence

import tensorflow as tf

from sknlp.layers import BiLSTM, AttentionPooling1D
from sknlp.module.text2vec import Text2vec
from .deep_retriever import DeepRetriever


class RNNRetriever(DeepRetriever):
    def __init__(
        self,
        classes: Sequence[str] = (0, 1),
        max_sequence_length: int = 100,
        projection_size: Optional[int] = None,
        temperature: float = 0.05,
        has_negative: bool = False,
        num_rnn_layers: int = 1,
        rnn_hidden_size: int = 512,
        rnn_dropout: float = 0.1,
        rnn_recurrent_dropout: float = 0.5,
        text2vec: Optional[Text2vec] = None,
        **kwargs
    ):
        super().__init__(
            classes,
            max_sequence_length=max_sequence_length,
            projection_size=projection_size,
            temperature=temperature,
            has_negative=has_negative,
            text2vec=text2vec,
            algorithm="rnn",
            **kwargs
        )
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_dropout = rnn_dropout
        self.rnn_recurrent_dropout = rnn_recurrent_dropout

    def build_encoding_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        embeddings = self.text2vec(inputs)
        mask = tf.math.not_equal(inputs, 0)
        return (
            BiLSTM(
                self.num_rnn_layers,
                self.rnn_hidden_size,
                dropout=self.rnn_dropout,
                recurrent_dropout=self.rnn_recurrent_dropout,
                return_sequences=True,
                name="bilstm",
            )(embeddings, mask),
            mask,
        )

    def build_intermediate_layer(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        encodings, mask = inputs
        if self.rnn_dropout:
            noise_shape = (None, 1, self.rnn_hidden_size * 2)
            encodings = tf.keras.layers.Dropout(
                self.rnn_dropout,
                noise_shape=noise_shape,
                name="encoding_dropout",
            )(encodings)
        return super().build_intermediate_layer(AttentionPooling1D()(encodings, mask))