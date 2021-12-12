from __future__ import annotations
from typing import Any, Sequence, Optional

import tensorflow as tf
from tensorflow.keras.layers import Dropout

from sknlp.layers import (
    BertPreprocessingLayer,
    BertAttentionMaskLayer,
)
from sknlp.data import BertRetrievalDataset, BertRetrievalEvaluationDataset
from sknlp.module.text2vec import Bert2vec
from .deep_retriever import DeepRetriever


class BertRetriever(DeepRetriever):
    dataset_class = BertRetrievalDataset
    evaluation_dataset_class = BertRetrievalEvaluationDataset

    def __init__(
        self,
        classes: Sequence[int] = (0, 1),
        max_sequence_length: int = 120,
        projection_size: Optional[int] = None,
        temperature: float = 0.05,
        has_negative: bool = False,
        cls_dropout: float = 0.1,
        text2vec: Optional[Bert2vec] = None,
        text_normalization: dict[str, str] = {"letter_case": "lowercase"},
        **kwargs
    ) -> None:
        super().__init__(
            classes,
            algorithm="bert",
            projection_size=projection_size,
            temperature=temperature,
            has_negative=has_negative,
            max_sequence_length=max_sequence_length,
            text2vec=text2vec,
            text_normalization=text_normalization,
            **kwargs
        )
        self.cls_dropout = cls_dropout
        self.inputs = [
            tf.keras.Input(shape=(None,), dtype=tf.int64, name="token_ids"),
            tf.keras.Input(shape=(None,), dtype=tf.int64, name="type_ids"),
        ]

    def build_encoding_layer(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        token_ids, type_ids = inputs
        mask = tf.math.not_equal(token_ids, 0)
        return [
            *self.text2vec(
                [token_ids, type_ids, BertAttentionMaskLayer()([type_ids, mask])]
            ),
            mask,
        ]

    def build_intermediate_layer(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        cls = inputs[0]
        if self.cls_dropout:
            cls = Dropout(self.cls_dropout, name="cls_dropout")(cls)
        return super().build_intermediate_layer(cls)

    def export(self, directory: str, name: str, version: str = "0") -> None:
        inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")
        preprocessing_layer = BertPreprocessingLayer(self.text2vec.vocab.sorted_tokens)
        token_ids, type_ids = preprocessing_layer(inputs)
        original_model = self._inference_model
        self._inference_model = tf.keras.Model(
            inputs=inputs, outputs=self._inference_model([token_ids, type_ids])
        )
        super().export(directory, name, version=version)
        self._inference_model = original_model