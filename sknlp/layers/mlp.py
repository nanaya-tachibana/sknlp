from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.keras.layers import Dense, BatchNormalization


class MLPLayer(Layer):

    def __init__(self, num_layers, hidden_size=256, output_size=1,
                 activation='tanh', name='mlp', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.input_spec = InputSpec(min_ndim=2)

        self.dense_layers = []
        self.batchnorm_layers = []
        for i in range(num_layers):
            if i == num_layers - 1:
                self.dense_layers.append(Dense(output_size,
                                               name='dense-%d' % i))
            else:
                _activation = activation if i == num_layers - 2 else 'relu'
                self.dense_layers.append(Dense(hidden_size,
                                               activation=_activation,
                                               name='dense-%d' % i))
                self.batchnorm_layers.append(BatchNormalization(momentum=0.9,
                                                                epsilon=1e-5,
                                                                axis=-1))

    def build(self, input_shape):
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

    def call(self, inputs):
        outputs = inputs
        for dense, batchnorm in zip(self.dense_layers[:-1],
                                    self.batchnorm_layers):
            outputs = batchnorm(dense(outputs))
        return self.dense_layers[-1](outputs)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, '
                'but saw: %s' % input_shape
            )
        return input_shape[:-1].concatenate(self.output_size)

    def get_config(self):
        config = {
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'activation': self.activation
        }
        base_config = super().get_config()
        return dict(**base_config, **config)
