import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dropout

from .lstmp import LSTMP


class MultiLSTMP(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers,
                 units,
                 projection_size=100,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 projection_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 projection_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 projection_constraint=None,
                 bias_constraint=None,
                 input_dropout=0.,
                 recurrent_dropout=0.,
                 output_dropout=0.,
                 recurrent_clip=None,
                 projection_clip=None,
                 last_connection='last'):
        self.num_layers = num_layers
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
        self.output_dropout = output_dropout
        self.recurrent_clip = recurrent_clip
        self.projection_clip = projection_clip
        self.last_connection = last_connection

        self.layers = []
        for i in range(self.num_layers):
            return_sequences = (
                i != self.num_layers - 1
                or self.last_connection != 'last'
            )
            self.layers.append(Bidirectional(LSTMP(
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
                return_sequences=return_sequences
            )))
        noise_shape = (None, 1, self.projection_size)
        if self.last_connection == 'last':
            noise_shape = None
        self.dropout_layer = Dropout(self.output_dropout,
                                     noise_shape=noise_shape)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        for layer in self.layers:
            inputs = layer(inputs, mask=mask, training=training,
                           initial_state=initial_state)
        return self.dropout_layer(inputs)

    def compute_mask(self, inputs, mask=None):
        return self.layers[0].compute_mask(inputs, mask)
