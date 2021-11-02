from __future__ import annotations
from typing import Any, Sequence, Optional

import tensorflow as tf
from tensorflow.keras.layers import Dropout

from sknlp.layers import (
    BertPreprocessingLayer,
    BertAttentionMaskLayer,
)
from sknlp.data import BertClassificationDataset
from sknlp.module.text2vec import Bert2vec
from .deep_classifier import DeepClassifier


class BertClassifier(DeepClassifier):
    dataset_class = BertClassificationDataset

    def __init__(
        self,
        classes: Sequence[str],
        is_multilabel: bool = True,
        is_pair_text: bool = False,
        max_sequence_length: int = 120,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 256,
        fc_activation: str = "tanh",
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        text2vec: Optional[Bert2vec] = None,
        text_normalization: dict[str, str] = {"letter_case": "lowercase"},
        **kwargs
    ) -> None:
        super().__init__(
            classes,
            is_multilabel=is_multilabel,
            is_pair_text=is_pair_text,
            algorithm="bert",
            num_fc_layers=num_fc_layers,
            fc_hidden_size=fc_hidden_size,
            fc_activation=fc_activation,
            max_sequence_length=max_sequence_length,
            text2vec=text2vec,
            text_normalization=text_normalization,
            **kwargs
        )
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.inputs = [
            tf.keras.Input(shape=(None,), dtype=tf.int64, name="token_ids"),
            tf.keras.Input(shape=(None,), dtype=tf.int64, name="type_ids"),
        ]

    def build_encoding_layer(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        self.text2vec.update_dropout(
            self.dropout, attention_dropout=self.attention_dropout
        )
        token_ids, type_ids = inputs
        mask = tf.math.not_equal(token_ids, 0)
        return [
            *self.text2vec(
                [token_ids, type_ids, BertAttentionMaskLayer()([type_ids, mask])]
            ),
            mask,
        ]

    def build_intermediate_layer(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        cls = inputs[0]
        if self.dropout:
            cls = Dropout(self.dropout, name="cls_dropout")(cls)
        return cls

    def export(self, directory: str, name: str, version: str = "0") -> None:
        if self.is_pair_text:
            inputs = [
                tf.keras.Input(shape=(), dtype=tf.string, name="text_input"),
                tf.keras.Input(shape=(), dtype=tf.string, name="context_input"),
            ]
        else:
            inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")
        preprocessing_layer = BertPreprocessingLayer(self.text2vec.vocab.sorted_tokens)
        token_ids, type_ids = preprocessing_layer(inputs)
        original_model = self._inference_model
        self._inference_model = tf.keras.Model(
            inputs=inputs, outputs=self._inference_model([token_ids, type_ids])
        )
        super().export(directory, name, version=version)
        self._inference_model = original_model

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "dropout": self.dropout}