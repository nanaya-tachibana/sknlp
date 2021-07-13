from __future__ import annotations
from typing import Sequence, Any, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from sknlp.data import ClassificationDataset
from sknlp.layers import MLPLayer
from sknlp.losses import MultiLabelCategoricalCrossentropy
from sknlp.metrics import PrecisionWithLogits, RecallWithLogits, FBetaScoreWithLogits
from sknlp.utils.classification import (
    logits2probabilities,
    probabilities2classes,
    classification_fscore,
)

from ..supervised_model import SupervisedNLPModel
from ..text2vec import Text2vec


class DeepClassifier(SupervisedNLPModel):
    dataset_class = ClassificationDataset
    dataset_args = ["is_multilabel"]

    def __init__(
        self,
        classes: Sequence[str],
        is_multilabel: bool = True,
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
        if "NULL" != classes[0]:
            if "NULL" in classes:
                classes.remove("NULL")
            classes.insert(0, "NULL")
        super().__init__(
            classes,
            max_sequence_length=max_sequence_length,
            text2vec=text2vec,
            task="classification",
            **kwargs,
        )
        self._is_multilabel = is_multilabel
        self._loss = loss
        self._loss_kwargs = loss_kwargs
        self.num_fc_layers = num_fc_layers
        self.fc_hidden_size = fc_hidden_size
        self.fc_activation = fc_activation

    @property
    def is_multilabel(self) -> bool:
        return self._is_multilabel

    @property
    def thresholds(self) -> list[float]:
        _thresholds = [0.5 for _ in range(self.num_classes)]
        _kthresholds: Optional[dict[str, float]] = self.prediction_kwargs.get(
            "thresholds", None
        )
        if _kthresholds is not None:
            for c, t in _kthresholds.items():
                _thresholds[self.class2idx(c)] = t
        return _thresholds

    @thresholds.setter
    def thresholds(self, thresholds: list[float]) -> None:
        if len(thresholds) != self.num_classes:
            raise ValueError(f"类别数为{self.num_classes}, 但thresholds长度为{len(thresholds)}")
        self.prediction_kwargs["thresholds"] = dict(zip(self.classes, thresholds))

    def get_loss(self, *args, **kwargs) -> list[tf.keras.losses.Loss]:
        if self.is_multilabel:
            if self._loss == "binary_crossentropy":
                return tf.keras.losses.BinaryCrossentropy(from_logits=True)
            return MultiLabelCategoricalCrossentropy()
        else:
            return tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def get_metrics(self, *args, **kwargs) -> list[tf.keras.metrics.Metric]:
        activation = "sigmoid" if self.is_multilabel else "softmax"
        class_id = None
        if self.num_classes == 2:
            class_id = 1
        return [
            PrecisionWithLogits(activation=activation, class_id=class_id),
            RecallWithLogits(activation=activation, class_id=class_id),
            FBetaScoreWithLogits(
                self.num_classes, activation=activation, class_id=class_id
            ),
        ]

    @classmethod
    def get_monitor(cls) -> str:
        return "val_fbeta_score"

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
        X: Sequence[str] = None,
        *,
        dataset: ClassificationDataset = None,
        batch_size: int = 128,
    ) -> np.ndarray:
        logits = super().predict(X=X, dataset=dataset, batch_size=batch_size)
        return logits2probabilities(logits, self.is_multilabel)

    def predict(
        self,
        X: Sequence[str] = None,
        *,
        dataset: ClassificationDataset = None,
        thresholds: Union[float, list[float], None] = None,
        batch_size: int = 128,
    ) -> Union[list[str], list[list[str]]]:
        probabilities = self.predict_proba(X=X, dataset=dataset, batch_size=batch_size)
        predictions = probabilities2classes(
            probabilities, self.is_multilabel, thresholds=thresholds or self.thresholds
        )
        if self.is_multilabel:
            return [
                [self.idx2class(i) for i in prediction] for prediction in predictions
            ]
        else:
            return [self.idx2class(i) for i in predictions if self.idx2class(i)]

    def score(
        self,
        X: Sequence[str] = None,
        y: Union[Sequence[Sequence[str]], Sequence[str]] = None,
        *,
        dataset: ClassificationDataset = None,
        thresholds: Union[float, list[float], None] = None,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        dataset = self.prepare_dataset(X, y, dataset)
        predictions = self.predict(
            dataset=dataset, thresholds=thresholds, batch_size=batch_size
        )
        return classification_fscore(
            dataset.y,
            predictions,
            self.classes[1:] if self.num_classes > 2 else self.classes,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "is_multilabel": self.is_multilabel,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DeepClassifier":
        config.pop("task", None)
        config.pop("algorithm", None)
        return super().from_config(config)