from __future__ import annotations
from typing import Any, Sequence, Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import Dropout

from sknlp.layers import (
    BertPreprocessingLayer,
    BertAttentionMaskLayer,
)
from sknlp.data import BertSimilarityDataset
from sknlp.module.text2vec import Bert2vec
from .deep_similarity import DeepSimilarity


class BertSimilarity(DeepSimilarity):
    dataset_class = BertSimilarityDataset

    def __init__(
        self,
        classes: Sequence[int] = (0, 1),
        max_sequence_length: int = 120,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 256,
        fc_activation: str = "tanh",
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        text2vec: Optional[Bert2vec] = None,
        **kwargs
    ) -> None:
        super().__init__(
            classes,
            algorithm="bert",
            num_fc_layers=num_fc_layers,
            fc_hidden_size=fc_hidden_size,
            fc_activation=fc_activation,
            max_sequence_length=max_sequence_length,
            text2vec=text2vec,
            **kwargs
        )
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")

    def build_preprocessing_layer(
        self, inputs: Union[tf.Tensor, list[tf.Tensor]]
    ) -> Union[tf.Tensor, list[tf.Tensor]]:
        return BertPreprocessingLayer(self.text2vec.vocab.sorted_tokens)(inputs)

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

    def build_intermediate_layer(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        cls = tf.keras.layers.Lambda(lambda x: x[0], name="cls_output")(inputs)
        if self.dropout:
            cls = Dropout(self.dropout, name="cls_dropout")(cls)
        return super().build_intermediate_layer(cls)

    def build_inference_model(self) -> tf.keras.Model:
        return tf.keras.Model(
            inputs=self._model.inputs,
            outputs=self._model.get_layer("cls_output").output,
        )

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "dropout": self.dropout}