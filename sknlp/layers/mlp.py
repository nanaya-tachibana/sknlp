from typing import Dict, Any, Optional, Callable
from itertools import zip_longest

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, InputSpec
from official.modeling import activations


def get_activation(activation_string: str) -> Callable:
    if activation_string == "gelu":
        return activations.gelu
    else:
        return tf.keras.activations.get(activation_string)


class MLPLayer(Layer):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int = 256,
        output_size: int = 1,
        activation: str = "relu",
        last_activation: Optional[str] = None,
        batch_normal: bool = True,
        momentum: float = 0.9,
        epsilon: float = 1e-5,
        name: str = "mlp",
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.last_activation = last_activation or activation
        self.batch_normal = batch_normal
        self.momentum = momentum
        self.epsilon = epsilon
        self.input_spec = InputSpec(min_ndim=2)

        self.dense_layers = []
        self.batchnorm_layers = []
        for i in range(num_layers):
            if i == num_layers - 1:
                self.dense_layers.append(Dense(output_size, name="dense-%d" % i))
            else:
                _activation = (
                    self.last_activation if i == num_layers - 2 else self.activation
                )
                self.dense_layers.append(
                    Dense(
                        hidden_size,
                        activation=get_activation(_activation),
                        name="dense-%d" % i,
                    )
                )
                if self.batch_normal:
                    self.batchnorm_layers.append(
                        BatchNormalization(axis=-1, momentum=momentum, epsilon=epsilon)
                    )

    def build(self, input_shape: tf.TensorShape) -> None:
        last_dim = input_shape[-1]
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs = inputs
        for dense, batchnorm in zip_longest(
            self.dense_layers[:-1], self.batchnorm_layers
        ):
            outputs = dense(outputs)
            if batchnorm is not None:
                outputs = batchnorm(outputs)
        return self.dense_layers[-1](outputs)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1] is None:
            raise ValueError(
                "The innermost dimension of input_shape must be defined, "
                "but saw: %s" % input_shape
            )
        return input_shape[:-1].concatenate(self.output_size)

    def get_config(self) -> Dict[str, Any]:
        config = {
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "activation": self.activation,
            "last_activation": self.last_activation,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
        }
        base_config = super().get_config()
        return dict(**base_config, **config)
