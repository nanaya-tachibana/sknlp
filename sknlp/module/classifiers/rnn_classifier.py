from __future__ import annotations

from tensorflow.keras.layers import Bidirectional, Dropout

from sknlp.layers import LSTMP
from .deep_classifier import DeepClassifier


class TextRNNClassifier(DeepClassifier):
    def __init__(
        self,
        classes: list[str],
        is_multilabel: bool = True,
        max_sequence_length: int = 100,
        dropout: float = 0.5,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 128,
        fc_activation: str = "tanh",
        num_rnn_layers: int = 1,
        rnn_hidden_size: int = 512,
        rnn_projection_size: int = 128,
        rnn_recurrent_clip: float = 3.0,
        rnn_projection_clip: float = 3.0,
        rnn_kernel_initializer="glorot_uniform",
        rnn_recurrent_initializer="orthogonal",
        rnn_projection_initializer="glorot_uniform",
        rnn_bias_initializer="zeros",
        rnn_kernel_regularizer=None,
        rnn_recurrent_regularizer=None,
        rnn_projection_regularizer=None,
        rnn_bias_regularizer=None,
        rnn_kernel_constraint=None,
        rnn_recurrent_constraint=None,
        rnn_projection_constraint=None,
        rnn_bias_constraint=None,
        text2vec=None,
        **kwargs
    ):
        super().__init__(
            classes,
            is_multilabel=is_multilabel,
            max_sequence_length=max_sequence_length,
            num_fc_layers=num_fc_layers,
            fc_hidden_size=fc_hidden_size,
            fc_activation=fc_activation,
            text2vec=text2vec,
            algorithm="rnn",
            **kwargs
        )
        self.dropout = dropout
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_projection_size = rnn_projection_size
        self.rnn_recurrent_clip = rnn_recurrent_clip
        self.rnn_projection_clip = rnn_projection_clip
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

    def build_encode_layer(self, inputs):
        outputs = self.text2vec(inputs)
        for i in range(self.num_rnn_layers):
            return_sequences = i != self.num_rnn_layers - 1
            outputs = Bidirectional(
                LSTMP(
                    self.rnn_hidden_size,
                    projection_size=self.rnn_projection_size,
                    recurrent_clip=self.rnn_recurrent_clip,
                    projection_clip=self.rnn_projection_clip,
                    dropout=self.dropout,
                    recurrent_dropout=self.dropout,
                    kernel_initializer=self.rnn_kernel_initializer,
                    recurrent_initializer=self.rnn_recurrent_initializer,
                    projection_initializer=self.rnn_projection_initializer,
                    bias_initializer=self.rnn_bias_initializer,
                    kernel_regularizer=self.rnn_kernel_regularizer,
                    recurrent_regularizer=self.rnn_recurrent_regularizer,
                    projection_regularizer=self.rnn_projection_regularizer,
                    bias_regularizer=self.rnn_bias_regularizer,
                    kernel_constraint=self.rnn_kernel_constraint,
                    recurrent_constraint=self.rnn_recurrent_constraint,
                    projection_constraint=self.rnn_projection_constraint,
                    bias_constraint=self.rnn_bias_constraint,
                    return_sequences=return_sequences,
                )
            )(outputs)
        if self.dropout:
            outputs = Dropout(self.dropout, noise_shape=None, name="encoder_dropout")(
                outputs
            )
        return outputs