from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer, Bidirectional, Dropout

from sknlp.typing import WeightRegularizer, WeightInitializer, WeightConstraint

from .lstmp import LSTMP


@tf.keras.utils.register_keras_serializable(package="sknlp")
class MultiLSTMP(Layer):
    def __init__(
        self,
        num_layers: int,
        units: int = 200,
        projection_size: int = 100,
        kernel_initializer: WeightInitializer = "glorot_uniform",
        recurrent_initializer: WeightInitializer = "orthogonal",
        projection_initializer: WeightInitializer = "glorot_uniform",
        bias_initializer: WeightInitializer = "zeros",
        kernel_regularizer: Optional[WeightRegularizer] = None,
        recurrent_regularizer: Optional[WeightRegularizer] = None,
        projection_regularizer: Optional[WeightRegularizer] = None,
        bias_regularizer: Optional[WeightRegularizer] = None,
        kernel_constraint: Optional[WeightConstraint] = None,
        recurrent_constraint: Optional[WeightConstraint] = None,
        projection_constraint: Optional[WeightConstraint] = None,
        bias_constraint: Optional[WeightConstraint] = None,
        input_dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        output_dropout: float = 0.0,
        recurrent_clip: Optional[float] = None,
        projection_clip: Optional[float] = None,
        last_connection: str = "last",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.last_connection = last_connection
        self.output_dropout = output_dropout
        self.units = units
        self.projection_size = projection_size
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.projection_initializer = projection_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.projection_regularizer = projection_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.recurrent_constraint = recurrent_constraint
        self.projection_constraint = projection_constraint
        self.bias_constraint = bias_constraint
        self.input_dropout = input_dropout
        self.recurrent_dropout = recurrent_dropout
        self.recurrent_clip = recurrent_clip
        self.projection_clip = projection_clip

    def build(self, input_shape: tf.TensorShape) -> None:
        self.layers = []
        for i in range(self.num_layers):
            return_sequences = (
                i != self.num_layers - 1 or self.last_connection != "last"
            )
            self.layers.append(
                Bidirectional(
                    LSTMP(
                        self.units,
                        self.projection_size,
                        kernel_initializer=self.kernel_initializer,
                        recurrent_initializer=self.recurrent_initializer,
                        projection_initializer=self.projection_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        recurrent_regularizer=self.recurrent_regularizer,
                        projection_regularizer=self.projection_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        recurrent_constraint=self.recurrent_constraint,
                        projection_constraint=self.projection_constraint,
                        bias_constraint=self.bias_constraint,
                        dropout=self.input_dropout,
                        recurrent_dropout=self.recurrent_dropout,
                        recurrent_clip=self.recurrent_clip,
                        projection_clip=self.projection_clip,
                        return_sequences=return_sequences,
                    )
                )
            )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        for layer in self.layers:
            inputs = layer(inputs, mask=mask)
        if self.output_dropout:
            noise_shape = (None, 1, self.layers[0].forward_layer.projection_size * 2)
            if self.last_connection == "last":
                noise_shape = None
            return Dropout(
                self.output_dropout, noise_shape=noise_shape, name="output_dropout"
            )(inputs)
        else:
            return inputs

    def get_config(self):
        return {
            **super().get_config(),
            "num_layers": self.num_layers,
            "units": self.layers[0].forward_layer.units,
            "projection_size": self.layers[0].forward_layer.projection_size,
            "kernel_initializer": self.layers[0].forward_layer.kernel_initializer,
            "recurrent_initializer": self.layers[0].forward_layer.recurrent_initializer,
            "projection_initializer": self.layers[
                0
            ].forward_layer.projection_initializer,
            "bias_initializer": self.layers[0].forward_layer.bias_initializer,
            "kernel_regularizer": self.layers[0].forward_layer.kernel_regularizer,
            "recurrent_regularizer": self.layers[0].forward_layer.recurrent_regularizer,
            "projection_regularizer": self.layers[
                0
            ].forward_layer.projection_regularizer,
            "bias_regularizer": self.layers[0].forward_layer.bias_regularizer,
            "kernel_constraint": self.layers[0].forward_layer.kernel_constraint,
            "recurrent_constraint": self.layers[0].forward_layer.recurrent_constraint,
            "projection_constraint": self.layers[0].forward_layer.projection_constraint,
            "bias_constraint": self.layers[0].forward_layer.bias_constraint,
            "input_dropout": self.layers[0].forward_layer.dropout,
            "recurrent_dropout": self.layers[0].forward_layer.recurrent_dropout,
            "recurrent_clip": self.layers[0].forward_layer.recurrent_clip,
            "projection_clip": self.layers[0].forward_layer.projection_clip,
            "output_dropout": self.output_dropout,
            "last_connection": self.last_connection,
        }