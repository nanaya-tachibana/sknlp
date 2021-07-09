from __future__ import annotations
from typing import Sequence, List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from tabulate import tabulate
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
    def __init__(
        self,
        classes: Sequence[str],
        is_multilabel: bool = True,
        max_sequence_length: Optional[int] = None,
        sequence_length: Optional[int] = None,
        segmenter: Optional[str] = "jieba",
        embedding_size: int = 100,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 256,
        fc_activation: str = "tanh",
        text2vec: Optional[Text2vec] = None,
        loss: Optional[str] = None,
        loss_kwargs: Optional[Dict[str, Any]] = None,
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
            sequence_length=sequence_length,
            segmenter=segmenter,
            embedding_size=embedding_size,
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

    def get_metrics(self, *args, **kwargs) -> List[tf.keras.metrics.Metric]:
        activation = "sigmoid" if self.is_multilabel else "softmax"
        return [
            PrecisionWithLogits(activation=activation),
            RecallWithLogits(activation=activation),
            FBetaScoreWithLogits(self.num_classes, activation=activation),
        ]

    @classmethod
    def get_monitor(cls) -> str:
        return "val_fbeta_score"

    def create_dataset_from_df(
        self, df: pd.DataFrame, no_label: bool = False
    ) -> ClassificationDataset:
        return ClassificationDataset(
            self.text2vec.vocab,
            self.classes,
            df=df,
            is_multilabel=self.is_multilabel,
            max_length=self.max_sequence_length,
            no_label=no_label,
            text_segmenter=self.text2vec.segmenter,
        )

    def create_dataset_from_csv(
        self, filename: str, no_label: bool = False
    ) -> ClassificationDataset:
        return ClassificationDataset(
            self.text2vec.vocab,
            self.classes,
            csv_file=filename,
            is_multilabel=self.is_multilabel,
            max_length=self.max_sequence_length,
            no_label=no_label,
            text_segmenter=self.text2vec.segmenter,
        )

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
        thresholds: Union[float, List[float], None] = None,
        batch_size: int = 128,
    ) -> Union[List[str], List[List[str]]]:
        probabilities = self.predict_proba(X=X, dataset=dataset, batch_size=batch_size)
        predictions = probabilities2classes(
            probabilities, self.is_multilabel, thresholds=thresholds or self.thresholds
        )
        if self.is_multilabel:
            return [
                [self.idx2class(i) for i in prediction if self.idx2class(i) != "NULL"]
                for prediction in predictions
            ]
        else:
            return [
                self.idx2class(i) if self.idx2class(i) != "NULL" else ""
                for i in predictions
                if self.idx2class(i)
            ]

    def score(
        self,
        X: Sequence[str] = None,
        y: Union[Sequence[Sequence[str]], Sequence[str]] = None,
        *,
        dataset: ClassificationDataset = None,
        thresholds: Union[float, List[float], None] = None,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        dataset = self.prepare_dataset(X, y, dataset)
        predictions = self.predict(
            dataset=dataset, thresholds=thresholds, batch_size=batch_size
        )
        return classification_fscore(dataset.y, predictions, classes=self.classes[1:])

    @classmethod
    def format_score(cls, score_df: pd.DataFrame, format: str = "markdown") -> str:
        return tabulate(score_df, headers="keys", tablefmt="github", showindex=False)

    def get_config(self) -> Dict[str, Any]:
        return {
            **super().get_config(),
            "is_multilabel": self.is_multilabel,
        }

    @classmethod
    def _filter_config(cls, config):
        config = super()._filter_config(config)
        config.pop("algorithm", None)
        config.pop("task", None)
        return config

    @classmethod
    def get_custom_objects(cls) -> Dict[str, Any]:
        return {
            **super().get_custom_objects(),
            "MLPLayer": MLPLayer,
            "MultiLabelCategoricalCrossentropy": MultiLabelCategoricalCrossentropy,
            "PrecisionWithLogits": PrecisionWithLogits,
            "RecallWithLogits": RecallWithLogits,
            "FBetaScoreWithLogits": FBetaScoreWithLogits,
        }