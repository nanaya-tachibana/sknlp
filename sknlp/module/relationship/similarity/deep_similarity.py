from __future__ import annotations
from typing import Sequence, Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from sknlp.data import SimilarityDataset
from sknlp.layers import MLPLayer
from sknlp.metrics import PrecisionWithLogits, RecallWithLogits, FBetaScoreWithLogits
from sknlp.utils.classification import classification_fscore
from sknlp.module.supervised_model import SupervisedNLPModel
from sknlp.module.text2vec import Text2vec


class DeepSimilarity(SupervisedNLPModel):
    dataset_class = SimilarityDataset

    def __init__(
        self,
        classes: Sequence[int] = (0, 1),
        max_sequence_length: Optional[int] = None,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 256,
        fc_activation: str = "tanh",
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
            task="similarity",
            **kwargs,
        )
        self._loss = loss
        self._loss_kwargs = loss_kwargs
        self.num_fc_layers = num_fc_layers
        self.fc_hidden_size = fc_hidden_size
        self.fc_activation = fc_activation

    def get_loss(self, *args, **kwargs) -> list[tf.keras.losses.Loss]:
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def get_metrics(self, *args, **kwargs) -> list[tf.keras.metrics.Metric]:
        activation = "sigmoid"
        return [
            PrecisionWithLogits(activation=activation),
            RecallWithLogits(activation=activation),
            FBetaScoreWithLogits(self.num_classes, activation=activation),
        ]

    def get_monitor(cls) -> str:
        return "val_fbeta_score"

    def build_intermediate_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        t1, t2 = tf.split(inputs, 2, axis=0)
        return tf.concat([t1, t2, tf.abs(t1 - t2)], -1)

    def build_output_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        logits = MLPLayer(
            self.num_fc_layers,
            hidden_size=self.fc_hidden_size,
            output_size=1,
            activation=self.fc_activation,
            name="mlp",
        )(inputs)
        return tf.squeeze(logits, axis=1)

    def predict(
        self,
        X: Sequence[tuple[str, str]] = None,
        *,
        dataset: SimilarityDataset = None,
        thresholds: Optional[float] = None,
        batch_size: int = 128,
    ) -> list[float]:
        thresholds = thresholds or 0.5
        predictions = super().predict(X=X, dataset=dataset, batch_size=batch_size)
        norm = np.linalg.norm(predictions, axis=-1, keepdims=True)
        t1, t2 = np.split(predictions / norm, 2, axis=0)
        return np.where((t1 * t2).sum(axis=-1) >= thresholds, 1.0, 0.0).tolist()

    def score(
        self,
        X: Sequence[tuple[str, str]] = None,
        y: Sequence[str] = None,
        *,
        dataset: SimilarityDataset = None,
        thresholds: Optional[float] = None,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        dataset = self.prepare_dataset(X, y, dataset)
        predictions = self.predict(
            dataset=dataset, thresholds=thresholds, batch_size=batch_size
        )
        return classification_fscore(dataset.y, predictions, self.classes)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DeepSimilarity":
        config.pop("task", None)
        config.pop("algorithm", None)
        return super().from_config(config)