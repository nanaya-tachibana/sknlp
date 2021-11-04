from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

from sknlp.layers import BiLSTM
from .rnn_classifier import RNNClassifier


class RCNNClassifier(RNNClassifier):
    def build_encoding_layer(self, inputs: tf.Tensor) -> list[tf.Tensor]:
        embeddings = self.text2vec(inputs)
        mask = tf.math.not_equal(inputs, 0)
        return [
            embeddings,
            BiLSTM(
                self.num_rnn_layers,
                self.rnn_hidden_size,
                dropout=self.dropout,
                recurrent_dropout=self.rnn_recurrent_dropout,
                return_sequences=True,
            )(embeddings, mask),
            mask,
        ]

    def build_intermediate_layer(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        embeddings, encodings, mask = inputs
        if self.dropout:
            noise_shape = (None, 1, self.text2vec.embedding_size)
            embeddings = Dropout(
                self.dropout,
                noise_shape=noise_shape,
                name="embedding_dropout",
            )(embeddings)

            noise_shape = (None, 1, self.rnn_hidden_size * 2)
            encodings = tf.keras.layers.Dropout(
                self.dropout,
                noise_shape=noise_shape,
                name="encoding_dropout",
            )(encodings)
        mixed_inputs = tf.concat([embeddings, encodings], axis=-1)
        mixed_outputs = Dense(self.fc_hidden_size, activation="tanh")(mixed_inputs)
        mask = tf.cast(mask, mixed_outputs.dtype)
        mask = tf.expand_dims(mask, -1)
        return tf.math.reduce_max(
            mixed_outputs * mask + mixed_outputs.dtype.min * (1 - mask), axis=1
        )
