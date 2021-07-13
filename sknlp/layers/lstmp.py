from typing import Optional

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import InputSpec
from tensorflow.python.keras.layers.recurrent import LSTM, LSTMCell
from tensorflow.python.training.tracking import data_structures

from sknlp.typing import WeightRegularizer, WeightInitializer, WeightConstraint


@tf.keras.utils.register_keras_serializable(package="sknlp")
class LSTMPCell(LSTMCell):
    """
    Long-Short Term Memory Projected (LSTMP) network cell with cell clip.
    (https://arxiv.org/abs/1402.1128)

    Parameters
    ----------
    units: Positive integer, dimensionality of the hidden state.
    projection_size: int, dimensionality of the output.
    activation: Activation function to use. Default: hyperbolic tangent
      (`tanh`). If you pass `None`, no activation is applied (ie. "linear"
      activation: `a(x) = x`).
    recurrent_activation: Activation function to use for the recurrent step.
      Default: sigmoid (`sigmoid`). If you pass `None`,
      no activation is applied (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix, used for
      the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel` weights
      matrix, used for the linear transformation of the recurrent state.
    projection_initializer: Initializer for the `projection_kernel` weights
      matrix, used for the linear transformation of the hidden state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at
      initialization. Setting it to true will also force
      `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
        al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    projection_regularizer: Regularizer function applied to
      the `projection_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix.
    projection_constraint: Constraint function applied to the
      `projection_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1. Fraction of the units to drop for the
      linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state.
    projection_dropout: Float between 0 and 1. Fraction of the units to drop
      for the linear transformation of the hidden state.
    implementation: Implementation mode, either 1 or 2.
      Mode 1 will structure its operations as a larger number of smaller dot
      products and additions, whereas mode 2 (default) will batch them into
      fewer, larger operations. These modes will have different performance
      profiles on different hardware and for different applications.

    Inputs
    ----------
    inputs: 2D tensor with shape: `(batch_size, input_length)`.
    states: List of state tensors corresponding to the previous timestep.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout`,
      `recurrent_dropout` or `projection_dropout`is used.
    """

    def __init__(
        self,
        units: int,
        projection_size: int = 100,
        activation: str = "tanh",
        recurrent_activation: str = "hard_sigmoid",
        use_bias: bool = True,
        kernel_initializer: WeightInitializer = "glorot_uniform",
        recurrent_initializer: WeightInitializer = "orthogonal",
        projection_initializer: WeightInitializer = "glorot_uniform",
        bias_initializer: WeightInitializer = "zeros",
        unit_forget_bias: bool = True,
        kernel_regularizer: Optional[WeightRegularizer] = None,
        recurrent_regularizer: Optional[WeightRegularizer] = None,
        projection_regularizer: Optional[WeightRegularizer] = None,
        bias_regularizer: Optional[WeightRegularizer] = None,
        kernel_constraint: Optional[WeightConstraint] = None,
        recurrent_constraint: Optional[WeightConstraint] = None,
        projection_constraint: Optional[WeightConstraint] = None,
        bias_constraint: Optional[WeightConstraint] = None,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        recurrent_clip: Optional[float] = None,
        projection_clip: Optional[float] = None,
        implementation: int = 2,
        **kwargs
    ) -> None:
        super().__init__(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation,
            **kwargs
        )
        self.projection_size = projection_size
        self.recurrent_clip = recurrent_clip
        self.projection_clip = projection_clip
        self.projection_initializer = initializers.get(projection_initializer)
        self.projection_regularizer = regularizers.get(projection_regularizer)
        self.projection_constraint = constraints.get(projection_constraint)
        self.state_size = data_structures.NoDependency(
            [self.projection_size, self.units]
        )
        self.output_size = self.projection_size

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.projection_size, self.units * 4),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate(
                        [
                            self.bias_initializer((self.units,), *args, **kwargs),
                            initializers.Ones()((self.units,), *args, **kwargs),
                            self.bias_initializer((self.units * 2,), *args, **kwargs),
                        ]
                    )

            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name="bias",
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        self.projection_kernel = self.add_weight(
            shape=(self.units, self.projection_size),
            name="projection_kernel",
            initializer=self.projection_initializer,
            regularizer=self.projection_regularizer,
            constraint=self.projection_constraint,
        )
        self.built = True

    def call(self, inputs, states, training=None):
        h, (_, c) = super().call(inputs, states, training=training)
        if self.recurrent_clip is not None:
            c = K.clip(c, -self.recurrent_clip, self.recurrent_clip)
        r = K.dot(h, self.projection_kernel)
        if self.projection_clip is not None:
            r = K.clip(r, -self.projection_clip, self.projection_clip)
        return r, [r, c]

    def get_config(self):
        config = {
            "projection_size": self.projection_size,
            "recurrent_clip": self.recurrent_clip,
            "projection_clip": self.projection_clip,
            "projection_initializer": initializers.serialize(
                self.projection_initializer
            ),
            "projection_regularizer": regularizers.serialize(
                self.projection_regularizer
            ),
            "projection_constraint": constraints.serialize(self.projection_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package="sknlp")
class LSTMP(LSTM):
    """

    Parameters
    ----------
    units: Positive integer, dimensionality of the hidden state.
    projection_size: int, dimensionality of the output.
    activation: Activation function to use. Default: hyperbolic tangent
      (`tanh`). If you pass `None`, no activation is applied (ie. "linear"
      activation: `a(x) = x`).
    recurrent_activation: Activation function to use for the recurrent step.
      Default: sigmoid (`sigmoid`). If you pass `None`,
      no activation is applied (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix, used for
      the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel` weights
      matrix, used for the linear transformation of the recurrent state.
    projection_initializer: Initializer for the `projection_kernel` weights
      matrix, used for the linear transformation of the hidden state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at
      initialization. Setting it to true will also force
      `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
        al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    projection_regularizer: Regularizer function applied to
      the `projection_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix.
    projection_constraint: Constraint function applied to the
      `projection_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1. Fraction of the units to drop for the
      linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state.
    projection_dropout: Float between 0 and 1. Fraction of the units to drop
      for the linear transformation of the hidden state.
    implementation: Implementation mode, either 1 or 2.
      Mode 1 will structure its operations as a larger number of smaller dot
      products and additions, whereas mode 2 (default) will batch them into
      fewer, larger operations. These modes will have different performance
      profiles on different hardware and for different applications.
    return_sequences: Boolean. Whether to return the last output.
      in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state
      in addition to the output.
    go_backwards: Boolean (default False).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default False).
      If True, the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `(timesteps, batch, ...)`, whereas in the False case, it will be
      `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.

    Inputs
    ----------
    inputs: 2D tensor with shape: `(batch_size, input_length)`.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout`,
      `recurrent_dropout` or `projection_dropout`is used.
    initial_states: List of state tensors corresponding to the
      previous timestep.
    """

    def __init__(
        self,
        units,
        projection_size: int = 100,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        use_bias: bool = True,
        kernel_initializer: WeightInitializer = "glorot_uniform",
        recurrent_initializer: WeightInitializer = "orthogonal",
        projection_initializer: WeightInitializer = "glorot_uniform",
        bias_initializer: WeightInitializer = "zeros",
        unit_forget_bias: bool = True,
        kernel_regularizer: Optional[WeightRegularizer] = None,
        recurrent_regularizer: Optional[WeightRegularizer] = None,
        projection_regularizer: Optional[WeightRegularizer] = None,
        bias_regularizer: Optional[WeightRegularizer] = None,
        kernel_constraint: Optional[WeightConstraint] = None,
        recurrent_constraint: Optional[WeightConstraint] = None,
        projection_constraint: Optional[WeightConstraint] = None,
        bias_constraint: Optional[WeightConstraint] = None,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        recurrent_clip: Optional[float] = None,
        projection_clip: Optional[float] = None,
        implementation: int = 2,
        activity_regularizer: Optional[WeightRegularizer] = None,
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: bool = False,
        stateful: bool = False,
        unroll: bool = False,
        **kwargs
    ):
        cell = LSTMPCell(
            units,
            projection_size=projection_size,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            projection_initializer=projection_initializer,
            unit_forget_bias=unit_forget_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            projection_regularizer=projection_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            projection_constraint=projection_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            recurrent_clip=recurrent_clip,
            projection_clip=projection_clip,
            implementation=implementation,
            dtype=kwargs.get("dtype"),
        )
        super(LSTM, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs
        )
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]

    @property
    def projection_size(self):
        return self.cell.projection_size

    @property
    def projection_initializer(self):
        return initializers.serialize(self.cell.projection_initializer)

    @property
    def projection_regularizer(self):
        return regularizers.serialize(self.cell.projection_regularizer)

    @property
    def projection_constraint(self):
        return constraints.serialize(self.cell.projection_constraint)

    @property
    def recurrent_clip(self):
        return self.cell.recurrent_clip

    @property
    def projection_clip(self):
        return self.cell.projection_clip

    def compute_mask(self, inputs, mask):
        output_mask = mask
        if self.return_state:
            state_mask = [None for _ in self.states]
            return [output_mask] + state_mask
        else:
            return output_mask

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, **self.cell.get_config()}

    @classmethod
    def from_config(cls, config):
        if "implementation" in config and config["implementation"] == 0:
            config["implementation"] = 2
        return cls(**config)