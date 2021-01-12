from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer, Bidirectional, Dropout

from sknlp.typing import WeightRegularizer, WeightInitializer, WeightConstraint

from .lstmp import LSTMP


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
        self.num_layers = num_layers
        self.last_connection = last_connection
        self.output_dropout = output_dropout

        self.layers = []
        for i in range(num_layers):
            return_sequences = i != num_layers - 1 or last_connection != "last"
            self.layers.append(
                Bidirectional(
                    LSTMP(
                        units,
                        projection_size,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        projection_initializer=projection_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        projection_regularizer=projection_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        projection_constraint=projection_constraint,
                        bias_constraint=bias_constraint,
                        dropout=input_dropout,
                        recurrent_dropout=recurrent_dropout,
                        recurrent_clip=recurrent_clip,
                        projection_clip=projection_clip,
                        return_sequences=return_sequences,
                    )
                )
            )
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
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
