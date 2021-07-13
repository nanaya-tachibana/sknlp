from __future__ import annotations
from typing import Sequence, List, Dict, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate
import tensorflow as tf
import tensorflow_addons as tfa

from sknlp.data import SimilarityDataset
from sknlp.layers import MLPLayer
from sknlp.metrics import BinaryAccuracyWithLogits
from sknlp.utils.classification import (
    classification_fscore,
    logits2probabilities,
    probabilities2classes,
)

from ..supervised_model import SupervisedNLPModel
from ..text2vec import Text2vec


class DeepDiscriminator(SupervisedNLPModel):
    dataset_class = SimilarityDataset

    def __init__(
        self,
        classes: Sequence[str] = ("相似度",),
        max_sequence_length: Optional[int] = None,
        sequence_length: Optional[int] = None,
        text2vec: Optional[Text2vec] = None,
        loss: Optional[str] = None,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            list(classes),
            max_sequence_length=max_sequence_length,
            sequence_length=sequence_length,
            text2vec=text2vec,
            task="similarity",
            **kwargs,
        )
        self._loss = loss
        self._loss_kwargs = loss_kwargs

    def get_loss(self, *args, **kwargs):
        if self._loss == "focal":
            return tfa.losses.SigmoidFocalCrossEntropy(
                from_logits=True, **self._loss_kwargs
            )
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def get_metrics(self, *args, **kwargs) -> List[tf.keras.metrics.Metric]:
        return [BinaryAccuracyWithLogits()]

    @classmethod
    def get_monitor(cls) -> str:
        return "val_binary_accuracy"

    def build_output_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        return MLPLayer(
            self.num_fc_layers,
            hidden_size=self.fc_hidden_size,
            output_size=self.num_classes,
            activation=self.fc_activation,
            name="mlp",
        )(inputs)

    def predict_proba(
        self,
        X: Sequence[Tuple[str, str]] = None,
        *,
        dataset: SimilarityDataset = None,
        batch_size: int = 128,
    ) -> np.ndarray:
        logits = super().predict(X=X, dataset=dataset, batch_size=batch_size)
        return logits2probabilities(logits, True)

    def predict(
        self,
        X: Sequence[Tuple[str, str]] = None,
        *,
        dataset: SimilarityDataset = None,
        thresholds: float = 0.5,
        batch_size: int = 128,
    ) -> List[float]:
        return self.predict_proba(X=X, dataset=dataset, batch_size=batch_size)

    def score(
        self,
        X: Sequence[Tuple[str, str]] = None,
        y: Sequence[float] = None,
        *,
        dataset: SimilarityDataset = None,
        thresholds: float = 0.5,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        dataset = self.prepare_dataset(X, y, dataset)
        probs = self.predict(dataset=dataset, batch_size=batch_size)
        predictions = probabilities2classes(probs, True, thresholds=thresholds)
        return classification_fscore(dataset.y, predictions, [0, 1])

    @classmethod
    def format_score(cls, score_df: pd.DataFrame, format: str = "markdown") -> str:
        return tabulate(score_df, headers="keys", tablefmt="github", showindex=False)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DeepDiscriminator":
        config.pop("task", None)
        config.pop("algorithm", None)
        return super().from_config(config)