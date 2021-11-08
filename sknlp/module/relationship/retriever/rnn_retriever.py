from __future__ import annotations
from typing import Optional, Sequence

import tensorflow as tf

from sknlp.typing import WeightInitializer
from sknlp.layers import BiLSTM
from sknlp.module.text2vec import Text2vec
from .deep_similarity import DeepSimilarity


class RNNSimilarity(DeepSimilarity):
    def __init__(
        self,
        classes: Sequence[str] = (0, 1),
        max_sequence_length: int = 100,
        dropout: float = 0.5,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 128,
        fc_activation: str = "tanh",
        num_rnn_layers: int = 1,
        rnn_hidden_size: int = 512,
        rnn_projection_size: int = 128,
        rnn_recurrent_dropout: float = 0.0,
        rnn_recurrent_clip: float = 3.0,
        rnn_projection_clip: float = 3.0,
        rnn_kernel_initializer: WeightInitializer = "glorot_uniform",
        rnn_recurrent_initializer: WeightInitializer = "orthogonal",
        rnn_projection_initializer: WeightInitializer = "glorot_uniform",
        text2vec: Optional[Text2vec] = None,
        **kwargs
    ):
        super().__init__(
            classes,
            max_sequence_length=max_sequence_length,
            num_fc_layers=num_fc_layers,
            fc_hidden_size=fc_hidden_size,
            fc_activation=fc_activation,
            text2vec=text2vec,
            algorithm="rnn",
            **kwargs
        )
        self.dropout = dropout
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_projection_size = rnn_projection_size
        self.rnn_recurrent_dropout = rnn_recurrent_dropout
        self.rnn_recurrent_clip = rnn_recurrent_clip
        self.rnn_projection_clip = rnn_projection_clip
        self.rnn_kernel_initializer = rnn_kernel_initializer
        self.rnn_recurrent_initializer = rnn_recurrent_initializer
        self.rnn_projection_initializer = rnn_projection_initializer

    def build_encoding_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        embeddings = self.text2vec(inputs)
        mask = tf.math.not_equal(inputs, 0)
        return BiLSTM(
            self.num_rnn_layers,
            self.rnn_hidden_size,
            projection_size=self.rnn_projection_size,
            recurrent_clip=self.rnn_recurrent_clip,
            projection_clip=self.rnn_projection_clip,
            dropout=self.dropout,
            recurrent_dropout=self.rnn_recurrent_dropout,
            kernel_initializer=self.rnn_kernel_initializer,
            recurrent_initializer=self.rnn_recurrent_initializer,
            projection_initializer=self.rnn_projection_initializer,
            return_sequences=False,
            name="bilstm",
        )(embeddings, mask)

    def build_inference_model(self) -> tf.keras.Model:
        return tf.keras.Model(
            inputs=self._model.inputs, outputs=self._model.get_layer("bilstm").output
        )