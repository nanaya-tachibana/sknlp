from __future__ import annotations
from typing import Any

import tensorflow as tf

from sknlp.layers import DilatedConvBlock
from sknlp.module.text2vec import Text2vec
from .deep_tagger import DeepTagger


class CNNTagger(DeepTagger):
    def __init__(
        self,
        classes: list[str],
        max_sequence_length: int = 100,
        output_format: str = "global_pointer",
        global_pointer_head_size: int = 64,
        crf_learning_rate_multiplier: float = 1.0,
        num_cnn_layers: int = 4,
        cnn_kernel_size: int = 3,
        cnn_max_dilation: int = 8,
        cnn_activation: str = "relu",
        cnn_dropout: float = 0.5,
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
            algorithm="cnn",
            **kwargs
        )
        self.cnn_dropout = cnn_dropout
        self.num_cnn_layers = num_cnn_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_max_dilation = cnn_max_dilation
        self.cnn_activation = cnn_activation
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
        mask_layer = tf.keras.layers.Lambda(
            lambda ids: tf.not_equal(ids, 0), name="mask_layer"
        )
        mask = mask_layer(token_ids)

        embeddings = self.text2vec(token_ids)
        outputs = [
            DilatedConvBlock(
                self.num_cnn_layers,
                kernel_size=self.cnn_kernel_size,
                max_dilation=self.cnn_max_dilation,
                dropout=self.cnn_dropout,
                activation=self.cnn_activation,
                return_sequences=True,
            )(embeddings, mask),
            mask,
        ]
        if self.output_format == "bio":
            outputs.append(tag_ids)
        return outputs

    def build_intermediate_layer(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        encodings = inputs[0]
        if self.cnn_dropout:
            noise_shape = (None, 1, self.text2vec.embedding_size)
            encodings = tf.keras.layers.Dropout(
                self.cnn_dropout,
                noise_shape=noise_shape,
                name="encoding_dropout",
            )(encodings)
        return [encodings, *inputs[1:]]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CNNTagger":
        config.pop("add_start_end_tag", None)
        return super().from_config(config)