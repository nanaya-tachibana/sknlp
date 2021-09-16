from __future__ import annotations
from typing import Optional

import tensorflow as tf

from sknlp.layers import DilatedConvBlock
from sknlp.module.text2vec import Text2vec
from .deep_classifier import DeepClassifier


class CNNClassifier(DeepClassifier):
    def __init__(
        self,
        classes: list[str],
        is_multilabel: bool = True,
        max_sequence_length: int = 100,
        num_cnn_layers: int = 5,
        cnn_kernel_size: int = 3,
        cnn_max_dilation: int = 16,
        cnn_activation: str = "tanh",
        dropout: float = 0.5,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 128,
        fc_activation: str = "tanh",
        text2vec: Optional[Text2vec] = None,
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
            algorithm="cnn",
            **kwargs
        )
        self.num_cnn_layers = num_cnn_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_max_dilation = cnn_max_dilation
        self.cnn_activation = cnn_activation
        self.dropout = dropout

    def build_encoding_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        embeddings = self.text2vec(inputs)
        mask = tf.math.not_equal(inputs, 0)
        return DilatedConvBlock(
            self.num_cnn_layers,
            kernel_size=self.cnn_kernel_size,
            max_dilation=self.cnn_max_dilation,
            dropout=self.dropout,
            activation=self.cnn_activation,
            return_sequences=False,
        )(embeddings, mask)
