from __future__ import annotations
from typing import Any
import tensorflow as tf
import numpy as np


class SinusoidalPositionEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        output_size: int,
        max_sequence_length: int,
        name: str = "sinusoidal_position_embedding",
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.output_size = output_size
        self.max_sequence_length = max_sequence_length

    def build(self, input_shape: tf.TensorShape) -> None:
        self.embedding = tf.keras.layers.Embedding(
            self.max_sequence_length,
            self.output_size,
        )
        self.embedding.build(input_shape)
        weights = np.array(
            [
                [
                    pos / np.power(10000, 2 * (j // 2) / self.output_size)
                    for j in range(self.output_size)
                ]
                for pos in range(self.max_sequence_length)
            ]
        )
        weights[:, ::2] = np.sin(weights[:, ::2])
        weights[:, 1::2] = np.cos(weights[:, 1::2])
        self.embedding.set_weights([weights])
        self.trainable = False
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> None:
        return self.embedding(inputs)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape([*input_shape.as_list(), self.output_size])

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "output_size": self.output_size,
            "max_sequence_length": self.max_sequence_length,
        }
