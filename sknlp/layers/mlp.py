from __future__ import annotations
from typing import Any

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, InputSpec


@tf.keras.utils.register_keras_serializable(package="sknlp")
class MLPLayer(Layer):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int = 256,
        output_size: int = 1,
        activation: str = "tanh",
        name: str = "mlp",
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.supports_masking = True

    def build(self, input_shape: tf.TensorShape) -> None:
        self.dense_layers = []
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                self.dense_layers.append(Dense(self.output_size, name="dense-%d" % i))
            else:
                self.dense_layers.append(
                    Dense(
                        self.hidden_size,
                        activation=tf.keras.activations.get(self.activation),
                        name="dense-%d" % i,
                    )
                )
        last_dim = input_shape[-1]
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs = inputs
        for dense in self.dense_layers:
            outputs = dense(outputs)
        return outputs

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1] is None:
            raise ValueError(
                "The innermost dimension of input_shape must be defined, "
                "but saw: %s" % input_shape
            )
        return input_shape[:-1].concatenate(self.output_size)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "activation": self.activation,
        }