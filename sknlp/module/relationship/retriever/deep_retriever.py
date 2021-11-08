from __future__ import annotations
from typing import Sequence, Any, Optional, Type

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

from sknlp.data import RetrievalDataset, ClassificationDataset
from sknlp.module.supervised_model import SupervisedNLPModel
from sknlp.module.text2vec import Text2vec


class DeepRetriever(SupervisedNLPModel):
    dataset_class = RetrievalDataset
    dataset_args = ["is_pair_text"]

    def __init__(
        self,
        classes: Sequence[int] = (0, 1),
        max_sequence_length: Optional[int] = None,
        projection_size: Optional[int] = None,
        has_negative: bool = False,
        text2vec: Optional[Text2vec] = None,
        loss: Optional[str] = None,
        loss_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        classes = list(classes)
        super().__init__(
            classes,
            max_sequence_length=max_sequence_length,
            text2vec=text2vec,
            task="retrieval",
            **kwargs,
        )
        self._loss = loss
        self._loss_kwargs = loss_kwargs
        self.projection_size = projection_size
        self.has_negative = has_negative

    @property
    def evaluation_dataset_class(self) -> Type[ClassificationDataset]:
        return ClassificationDataset

    @property
    def is_pair_text(self) -> bool:
        return True

    def get_loss(self, *args, **kwargs) -> list[tf.keras.losses.Loss]:
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def get_metrics(self, *args, **kwargs) -> list[tf.keras.metrics.Metric]:
        return []

    def get_monitor(cls) -> str:
        return None

    def build_intermediate_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        pooling_layer = tf.keras.layers.Lambda(lambda x: x, name="pooling")
        if self.projection_size is not None:
            pooling_layer = tf.keras.layers.Dense(self.projection_size, name="pooling")
        return pooling_layer(inputs)

    def build_output_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        num_inputs = 2 + self.has_negative
        normalized = inputs / tf.linalg.norm(inputs, axis=-1, keepdims=True)
        reshaped = tf.reshape(normalized, (-1, num_inputs, tf.shape(inputs)[-1]))
        input_list: list[tf.Tensor] = tf.split(reshaped, num_inputs, axis=1)

        input = tf.squeeze(input_list[0], axis=1)
        positive = tf.squeeze(input_list[1], axis=1)
        cos_sim = tf.matmul(input, positive, transpose_b=True)
        if num_inputs == 3:
            negative = tf.squeeze(input_list[2], axis=1)
            cos_sim = tf.concat(
                [cos_sim, tf.matmul(input, negative, transpose_b=True)], -1
            )
        return cos_sim * 20

    def build_inference_model(self) -> tf.keras.Model:
        return tf.keras.Model(
            inputs=self._model.inputs,
            outputs=self._model.get_layer("pooling").output,
        )

    def predict(
        self,
        X: Sequence[tuple[str, str]] = None,
        *,
        dataset: RetrievalDataset = None,
        batch_size: int = 128,
    ) -> list[float]:
        predictions = super().predict(X=X, dataset=dataset, batch_size=batch_size)
        normalized = predictions / np.linalg.norm(predictions, axis=-1, keepdims=True)
        return (normalized[::2, :] * normalized[1::2, :]).sum(axis=-1).tolist()

    def score(
        self,
        X: Sequence[tuple[str, str]] = None,
        y: Sequence[int] = None,
        *,
        dataset: RetrievalDataset = None,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        predictions = self.predict(X=X, dataset=dataset, batch_size=batch_size)
        spearman = stats.spearmanr(list(y), predictions).correlation
        return pd.DataFrame(("spearman", spearman), columns=["score", "value"])

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "projection_size": self.projection_size,
            "has_negative": self.has_negative,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DeepRetriever":
        config.pop("task", None)
        config.pop("algorithm", None)
        return super().from_config(config)