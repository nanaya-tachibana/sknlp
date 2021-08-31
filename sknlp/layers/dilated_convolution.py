from __future__ import annotations
from sknlp import activations
from typing import Optional, Any
import math

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, Dropout

from .attention import AttentionPooling1D


@tf.keras.utils.register_keras_serializable(package="sknlp")
class GatedDilatedConv1D(Layer):
    """
    实现参考: https://github.com/bojone/dgcnn_for_reading_comprehension/blob/master/dgcnn/dgcnn.py
    """

    def __init__(
        self,
        output_size: Optional[int] = None,
        kernel_size: int = 3,
        dilation_rate: int = 1,
        skip_connect: bool = True,
        dropout: float = 0.5,
        gate_dropout: float = 0.1,
        activation: str = "tanh",
        name: str = "dilated_conv1d",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.skip_connect = skip_connect
        self.dropout = dropout
        self.gate_dropout = gate_dropout
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape: tf.TensorShape) -> None:
        if self.output_size is None:
            self.output_size = input_shape[-1]

        kernel_initializer = "glorot_uniform"
        if self.activation is tf.keras.activations.relu:
            kernel_initializer = "he_uniform"
        self.conv1d = Conv1D(
            2 * self.output_size,
            self.kernel_size,
            kernel_initializer=kernel_initializer,
            activation=self.activation,
            padding="same",
            dilation_rate=self.dilation_rate,
        )
        self.conv1d_projection = None
        if self.skip_connect and self.output_size != input_shape[-1]:
            self.conv1d_projection = Conv1D(self.output_size, 1)

        if self.dropout:
            noise_shape = (None, 1, input_shape[-1])
            self.input_dropout_layer = Dropout(self.dropout, noise_shape=noise_shape)
        if self.gate_dropout:
            noise_shape = (None, 1, self.output_size)
            self.gate_dropout_layer = Dropout(
                self.gate_dropout, noise_shape=noise_shape
            )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        if self.dropout:
            inputs = self.input_dropout_layer(inputs)

        mask = tf.cast(mask, inputs.dtype)
        mask = tf.expand_dims(mask, -1)
        masked_inputs = inputs * mask
        conv_outputs, gate = tf.split(self.conv1d(masked_inputs), 2, axis=-1)
        gate = tf.sigmoid(gate)
        if self.gate_dropout:
            gate = self.gate_dropout_layer(gate)

        outputs = conv_outputs * mask * gate
        if self.skip_connect:
            if self.conv1d_projection is not None:
                masked_inputs = self.conv1d_projection(masked_inputs)
            outputs += masked_inputs * (1 - gate)

        return self.activation(outputs)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape([*input_shape[:-1], self.output_size])

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "output_size": self.output_size,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "skip_connect": self.skip_connect,
            "dropout": self.dropout,
            "gate_dropout": self.gate_dropout,
            "activation": tf.keras.activations.serialize(self.activation),
        }


@tf.keras.utils.register_keras_serializable(package="sknlp")
class DilatedConvBlock(Layer):
    def __init__(
        self,
        num_layers: int,
        output_size: Optional[int] = None,
        max_dilation: int = 16,
        kernel_size: int = 3,
        skip_connect: bool = True,
        dropout: float = 0.2,
        activation: str = "tanh",
        return_sequences: bool = True,
        name: str = "conv_block",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.output_size = output_size
        self.max_dilation = max_dilation
        self.kernel_size = kernel_size
        self.skip_connect = skip_connect
        self.dropout = dropout
        self.activation = activation
        self.return_sequences = return_sequences
        self.conv_layers: list[GatedDilatedConv1D] = []

    def build(self, input_shape: tf.TensorShape) -> None:
        if self.output_size is None:
            self.output_size = input_shape[-1]

        period = int(math.floor(math.log2(self.max_dilation))) + 1
        for i in range(self.num_layers):
            if i == self.num_layers - 1 and i % period == 0:
                break
            self.conv_layers.append(
                GatedDilatedConv1D(
                    output_size=self.output_size,
                    kernel_size=self.kernel_size,
                    dilation_rate=2 ** (i % period),
                    skip_connect=self.skip_connect,
                    dropout=self.dropout,
                    activation=self.activation,
                    name=f"conv1d/layer_{i}",
                )
            )
        self.conv_layers.append(
            GatedDilatedConv1D(
                output_size=self.output_size,
                kernel_size=self.kernel_size,
                dilation_rate=1,
                skip_connect=self.skip_connect,
                dropout=self.dropout,
                activation=None,
                name=f"conv1d/layer_{i + 1}",
            )
        )
        if not self.return_sequences:
            self.pooling = AttentionPooling1D(output_size=self.output_size)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        outputs = inputs
        for conv_layer in self.conv_layers:
            outputs = conv_layer(outputs, mask)
        if self.return_sequences:
            return outputs
        else:
            return self.pooling(outputs, mask)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        if self.return_sequences:
            return tf.TensorShape([*input_shape[:-1], self.output_size])
        return tf.TensorShape([input_shape[0], self.output_size])

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "kernel_size": self.kernel_size,
            "max_dilation": self.max_dilation,
            "skip_connect": self.skip_connect,
            "dropout": self.dropout,
            "activation": self.activation,
            "return_sequences": self.return_sequences,
        }
