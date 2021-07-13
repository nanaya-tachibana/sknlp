from __future__ import annotations
from typing import Any

import tensorflow as tf

from sknlp.layers import LSTMP
from .deep_tagger import DeepTagger


class TextRNNTagger(DeepTagger):
    def __init__(
        self,
        classes: list[str],
        max_sequence_length: int = 100,
        use_crf: bool = False,
        crf_learning_rate_multiplier: float = 1.0,
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
        dropout: float = 0.5,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 128,
        fc_activation: str = "tanh",
        text2vec=None,
        **kwargs
    ):
        super().__init__(
            classes,
            add_start_end_tag=False,
            use_crf=use_crf,
            crf_learning_rate_multiplier=crf_learning_rate_multiplier,
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
        self.inputs = [
            tf.keras.Input(shape=(None,), dtype=tf.int32, name="token_id"),
            tf.keras.Input(shape=(None,), dtype=tf.int32, name="tag_id"),
        ]

    def build_encode_layer(self, inputs):
        token_ids, tag_ids = inputs
        mask = tf.keras.layers.Lambda(
            lambda x: tf.cast(x != 0, tf.int32), name="mask_layer"
        )(token_ids)
        outputs = self.text2vec(token_ids)
        for _ in range(self.num_rnn_layers):
            outputs = tf.keras.layers.Bidirectional(
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
                    return_sequences=True,
                )
            )(outputs)
        if self.dropout:
            noise_shape = (None, 1, self.rnn_projection_size * 2)
            outputs = tf.keras.layers.Dropout(
                self.dropout, noise_shape=noise_shape, name="encoder_dropout"
            )(outputs)
        return outputs, mask, tag_ids

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "dropout": self.dropout}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TextRNNTagger":
        config.pop("add_start_end_tag", None)
        return super().from_config(config)