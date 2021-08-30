from __future__ import annotations
from typing import Any

import tensorflow as tf

from sknlp.typing import WeightInitializer
from sknlp.layers import BiLSTM
from sknlp.module.text2vec import Text2vec
from .deep_tagger import DeepTagger


class RNNTagger(DeepTagger):
    def __init__(
        self,
        classes: list[str],
        max_sequence_length: int = 100,
        output_format: str = "global_pointer",
        global_pointer_head_size: int = 64,
        crf_learning_rate_multiplier: float = 1.0,
        num_rnn_layers: int = 1,
        rnn_hidden_size: int = 512,
        rnn_projection_size: int = 128,
        rnn_recurrent_clip: float = 3.0,
        rnn_projection_clip: float = 3.0,
        rnn_kernel_initializer: WeightInitializer = "glorot_uniform",
        rnn_recurrent_initializer: WeightInitializer = "orthogonal",
        rnn_projection_initializer: WeightInitializer = "glorot_uniform",
        dropout: float = 0.5,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 128,
        fc_activation: str = "tanh",
        text2vec: Text2vec = None,
        **kwargs
    ):
        super().__init__(
            classes,
            add_start_end_tag=False,
            output_format=output_format,
            global_pointer_head_size=global_pointer_head_size,
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
        if self.output_format == "bio":
            self.inputs = [
                tf.keras.Input(shape=(None,), dtype=tf.int32, name="token_id"),
                tf.keras.Input(shape=(None,), dtype=tf.int32, name="tag_id"),
            ]
        else:
            self.inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="token_id")

    def build_encoding_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.output_format == "bio":
            token_ids, tag_ids = inputs
        else:
            token_ids = inputs
        mask = tf.keras.layers.Lambda(
            lambda x: tf.cast(x != 0, tf.int32), name="mask_layer"
        )(token_ids)

        embeddings = self.text2vec(token_ids)
        outputs = [
            BiLSTM(
                self.num_rnn_layers,
                self.rnn_hidden_size,
                projection_size=self.rnn_projection_size,
                recurrent_clip=self.rnn_recurrent_clip,
                projection_clip=self.rnn_projection_clip,
                dropout=self.dropout,
                kernel_initializer=self.rnn_kernel_initializer,
                recurrent_initializer=self.rnn_recurrent_initializer,
                projection_initializer=self.rnn_projection_initializer,
                return_sequences=True,
            )(embeddings, mask),
            mask,
        ]
        if self.output_format == "bio":
            outputs.append(tag_ids)
        return outputs

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "dropout": self.dropout}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RNNTagger":
        config.pop("add_start_end_tag", None)
        return super().from_config(config)