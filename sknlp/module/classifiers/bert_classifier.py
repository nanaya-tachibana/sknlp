from __future__ import annotations
from typing import Any, Sequence, Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import Dropout

from sknlp.layers import BertPreprocessingLayer, BertAttentionMaskLayer
from sknlp.data import BertClassificationDataset
from sknlp.module.text2vec import Bert2vec
from .deep_classifier import DeepClassifier


class BertClassifier(DeepClassifier):
    dataset_class = BertClassificationDataset

    def __init__(
        self,
        classes: Sequence[str],
        is_multilabel: bool = True,
        max_sequence_length: int = 120,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 256,
        fc_activation: str = "tanh",
        dropout: float = 0.5,
        text2vec: Optional[Bert2vec] = None,
        **kwargs
    ) -> None:
        super().__init__(
            classes,
            is_multilabel=is_multilabel,
            algorithm="bert",
            num_fc_layers=num_fc_layers,
            fc_hidden_size=fc_hidden_size,
            fc_activation=fc_activation,
            max_sequence_length=max_sequence_length,
            text2vec=text2vec,
            **kwargs
        )
        self.dropout = dropout
        self.inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")

    def build_preprocessing_layer(
        self, inputs: Union[tf.Tensor, list[tf.Tensor]]
    ) -> Union[tf.Tensor, list[tf.Tensor]]:
        layer = BertPreprocessingLayer(self.text2vec.vocab.sorted_tokens)
        return layer(inputs)

    def build_encoding_layer(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        if self.dropout:
            self.text2vec.update_dropout(dropout=self.dropout)
        mask = self.text2vec.compute_mask(inputs)
        token_ids, type_ids = inputs
        return (
            *self.text2vec(
                [token_ids, type_ids, BertAttentionMaskLayer()([type_ids, mask])]
            ),
            mask,
        )

    def build_intermediate_layer(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        cls = inputs[0]
        if self.dropout:
            cls = Dropout(self.dropout, name="cls_dropout")(cls)
        return cls

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "dropout": self.dropout}