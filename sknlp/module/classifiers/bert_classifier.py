from __future__ import annotations
from typing import Any, Sequence, Optional

import tensorflow as tf
from tensorflow.keras.layers import Dropout

from sknlp.layers import BertPreprocessingLayer
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

    def build_encode_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        preprocessing_layer = BertPreprocessingLayer(self.text2vec.vocab.sorted_tokens)
        token_ids = preprocessing_layer(inputs)
        if self.dropout:
            self.text2vec.update_dropout(dropout=self.dropout)
        _, cls = self.text2vec(
            [
                token_ids,
                tf.zeros_like(token_ids, dtype=tf.int64),
            ]
        )
        if self.dropout:
            cls = Dropout(self.dropout, name="cls_dropout")(cls)

        return cls

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "dropout": self.dropout}