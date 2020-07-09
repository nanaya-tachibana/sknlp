import tensorflow as tf


class TextRNNLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        num_rnn_layers=1,
        rnn_hidden_size=256,
        rnn_projection_size=128,
        rnn_recurrent_clip=3,
        rnn_projection_clip=3,
        rnn_input_dropout=0.5,
        rnn_recurrent_dropout=0.5,
        rnn_output_dropout=0.5,
        rnn_last_connection=None,
        rnn_kernel_initializer='glorot_uniform',
        rnn_recurrent_initializer='orthogonal',
        rnn_projection_initializer='glorot_uniform',
        rnn_bias_initializer='zeros',
        rnn_kernel_regularizer=None,
        rnn_recurrent_regularizer=None,
        rnn_projection_regularizer=None,
        rnn_bias_regularizer=None,
        rnn_kernel_constraint=None,
        rnn_recurrent_constraint=None,
        rnn_projection_constraint=None,
        rnn_bias_constraint=None,
    ):
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_last_connection = rnn_last_connection
        self.rnn_projection_size = rnn_projection_size
        self.rnn_recurrent_clip = rnn_recurrent_clip
        self.rnn_projection_clip = rnn_projection_clip
        self.rnn_input_dropout = rnn_input_dropout
        self.rnn_recurrent_dropout = rnn_recurrent_dropout
        self.rnn_output_dropout = rnn_output_dropout
        self.rnn_kernel_initializer = rnn_kernel_initializer
        self.rnn_recurrent_initializer = rnn_recurrent_initializer
        self.rnn_projection_initializer = rnn_projection_initializer
        self.rnn_bias_initializer = rnn_bias_initializer
        self.rnn_kernel_regularizer = rnn_kernel_regularizer
        self.rnn_recurrent_regularizer = rnn_recurrent_regularizer
        self.rnn_projection_regularizer = rnn_projection_regularizer
        self.rnn_bias_regularizer = rnn_bias_regularizer
        self.rnn_kernel_constraint = rnn_kernel_constraint
        self.rnn_recurrent_constraint = rnn_recurrent_constraint
        self.rnn_projection_constraint = rnn_projection_constraint
        self.rnn_bias_constraint = rnn_bias_constraint
