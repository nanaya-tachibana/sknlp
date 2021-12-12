from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM


@tf.keras.utils.register_keras_serializable(package="sknlp")
class BiLSTM(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        return_sequences: bool = False,
        first_layer_dropout: float | None = None,
        name: str = "bilstm",
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.return_sequences = return_sequences
        self.first_layer_dropout = first_layer_dropout

    def build(self, input_shape: tf.TensorShape) -> None:
        self.rnn_layers: list[Bidirectional] = []
        for i in range(self.num_layers):
            return_sequences = self.return_sequences or i != self.num_layers - 1
            dropout = self.dropout
            if i == 0 and self.first_layer_dropout is not None:
                dropout = self.first_layer_dropout
            self.rnn_layers.append(
                Bidirectional(
                    LSTM(
                        self.hidden_size,
                        dropout=dropout,
                        recurrent_dropout=self.recurrent_dropout,
                        return_sequences=return_sequences,
                    )
                )
            )
        super().build(input_shape)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        if self.return_sequences:
            return [*input_shape[:-1], self.hidden_size * 2]
        else:
            return [input_shape[0], self.hidden_size * 2]

    def call(self, inputs: tf.Tensor, mask: tf.Tensor, **kwargs) -> tf.Tensor:
        outputs = inputs
        for layer in self.rnn_layers:
            outputs = layer(outputs, mask=mask)
        return outputs

    def get_config(self):
        return {
            **super().get_config(),
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "return_sequences": self.return_sequences,
            "first_layer_dropout": self.first_layer_dropout,
        }