from __future__ import annotations
from typing import Optional, Union, Callable, Any

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


@tf.keras.utils.register_keras_serializable(package="sknlp")
class AttentionPooling1D(Layer):
    def __init__(
        self,
        output_size: Optional[int] = None,
        activation: Union[str, Callable] = "tanh",
        use_bias: bool = False,
        name: str = "attention_pooling",
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.output_size = output_size
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape: tf.TensorShape) -> None:
        if self.output_size is None:
            self.output_size = input_shape[-1]
        self.dense = Dense(
            self.output_size, activation=self.activation, use_bias=self.use_bias
        )
        self.key = Dense(1, use_bias=False)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        score = self.key(self.dense(inputs))
        mask = tf.cast(mask, score.dtype)
        mask = tf.expand_dims(mask, -1)
        score = score * mask + (1 - mask) * score.dtype.min
        return tf.reduce_sum(tf.math.softmax(score, axis=1) * inputs, axis=1)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape([input_shape[0], self.output_size])

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "output_size": self.output_size,
            "activation": tf.keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
        }