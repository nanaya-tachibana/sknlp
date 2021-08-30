from __future__ import annotations
from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

from sknlp.typing import WeightInitializer
from sknlp.layers import BiLSTM
from sknlp.module.text2vec import Text2vec
from .rnn_classifier import TextRNNClassifier


class TextRCNNClassifier(TextRNNClassifier):
    def build_encoding_layer(self, inputs: tf.Tensor) -> list[tf.Tensor]:
        embeddings = self.text2vec(inputs)
        mask = self.text2vec.compute_mask(inputs)
        return (
            embeddings,
            BiLSTM(
                self.num_rnn_layers,
                self.rnn_hidden_size,
                projection_size=self.rnn_projection_size,
                recurrent_clip=self.rnn_recurrent_clip,
                projection_clip=self.rnn_projection_clip,
                dropout=self.dropout,
                kernel_initializer=self.rnn_kernel_initializer,
                recurrent_initializer=self.rnn_recurrent_initializer,
                projection_initializer=self.rnn_projection_initializer,
                return_sequences=True,
            )(embeddings, mask),
        )

    def build_intermediate_layer(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        embeddings, encodings = inputs
        if self.dropout:
            noise_shape = (None, 1, self.text2vec.embedding_size)
            embeddings = Dropout(
                self.dropout,
                noise_shape=noise_shape,
                name="embedding_dropout",
            )(embeddings)
        mixed_inputs = tf.concat([embeddings, encodings], axis=-1)
        mixed_outputs = Dense(self.fc_hidden_size, activation="tanh")(mixed_inputs)
        return tf.math.reduce_max(mixed_outputs, axis=1)