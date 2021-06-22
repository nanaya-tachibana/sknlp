from typing import Sequence, List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from tabulate import tabulate
import tensorflow as tf

from sknlp.data import ClassificationDataset
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
        use_batch_normalization: bool = True,
        text2vec: Optional[Text2vec] = None,
        loss: Optional[str] = None,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
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
            **kwargs
        )
        self._is_multilabel = is_multilabel
        self._use_batch_normalization = use_batch_normalization
        self._loss = loss
        self._loss_kwargs = loss_kwargs

    @property
    def is_multilabel(self):
        return self._is_multilabel

    @property
    def use_batch_normalization(self):
        return self._use_batch_normalization

    def get_loss(self, *args, **kwargs):
        if self.is_multilabel:
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

    def predict_proba(
        self,
        X: Sequence[str] = None,
        *,
        dataset: ClassificationDataset = None,
        batch_size: int = 128
    ) -> np.ndarray:
        logits = super().predict(X=X, dataset=dataset, batch_size=batch_size)
        return logits2probabilities(logits, self.is_multilabel)

    def predict(
        self,
        X: Sequence[str] = None,
        *,
        dataset: ClassificationDataset = None,
        thresholds: Union[float, List[float]] = 0.5,
        batch_size: int = 128
    ) -> Union[List[str], List[List[str]]]:
        probabilities = self.predict_proba(X=X, dataset=dataset, batch_size=batch_size)
        predictions = probabilities2classes(
            probabilities, self.is_multilabel, thresholds=thresholds
        )
        if self.is_multilabel:
            return [
                [self.idx2class(i) for i in prediction if self.idx2class(i) != "NULL"]
                for prediction in predictions
            ]
        else:
            return [
                self.idx2class(i) for i in predictions if self.idx2class(i) != "NULL"
            ]

    def score(
        self,
        X: Sequence[str] = None,
        y: Union[Sequence[Sequence[str]], Sequence[str]] = None,
        *,
        dataset: ClassificationDataset = None,
        thresholds: Union[float, List[float]] = 0.5,
        batch_size: int = 128
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
            "use_batch_normalization": self.use_batch_normalization,
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
            "MultiLabelCategoricalCrossentropy": MultiLabelCategoricalCrossentropy,
            "PrecisionWithLogits": PrecisionWithLogits,
            "RecallWithLogits": RecallWithLogits,
            "FBetaScoreWithLogits": FBetaScoreWithLogits,
        }