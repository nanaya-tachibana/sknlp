from typing import Sequence, List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from tabulate import tabulate
import tensorflow as tf

from sknlp.vocab import Vocab
from sknlp.data import ClassificationDataset
from sknlp.metrics import PrecisionWithLogits, RecallWithLogits, FBetaScoreWithLogits

from ..supervised_model import SupervisedNLPModel
from ..text2vec import Text2vec

from .utils import logits2probabilities, probabilities2classes, classification_fscore


class DeepClassifier(SupervisedNLPModel):
    def __init__(
        self,
        classes: Sequence[str],
        is_multilabel: bool = True,
        max_sequence_length: Optional[int] = None,
        sequence_length: Optional[int] = None,
        segmenter: Optional[str] = "jieba",
        embedding_size: int = 100,
        text2vec: Optional[Text2vec] = None,
        loss: str = None,
        **kwargs
    ):
        classes = list(classes)
        if not is_multilabel and "NULL" != classes[0]:
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
        self._loss = loss

    @property
    def is_multilabel(self):
        return self._is_multilabel

    def get_loss(self, *args, **kwargs):
        if self.is_multilabel:
            return tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            return tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def get_callbacks(self, *args, **kwargs) -> List[tf.keras.callbacks.Callback]:
        return []

    def get_metrics(self, *args, **kwargs) -> List[tf.keras.metrics.Metric]:
        if self.is_multilabel:
            return [
                PrecisionWithLogits(),
                RecallWithLogits(),
                FBetaScoreWithLogits(self.num_classes),
            ]
        else:
            return [
                PrecisionWithLogits(logits2scores="softmax"),
                RecallWithLogits(logits2scores="softmax"),
                FBetaScoreWithLogits(self.num_classes, logits2scores="softmax"),
            ]

    def get_monitor(self) -> str:
        return "val_fbeta_score"

    def create_dataset_from_df(
        self, df: pd.DataFrame, vocab: Vocab, segmenter: str, labels: Sequence[str]
    ) -> ClassificationDataset:
        return ClassificationDataset(
            vocab,
            list(labels),
            df=df,
            is_multilabel=self.is_multilabel,
            max_length=self.max_sequence_length,
            text_segmenter=segmenter,
        )

    def dummy_y(self, X: Sequence[str]) -> List[str]:
        return ["O" for _ in range(len(X))]

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
                [self.idx2class(i) for i in prediction] for prediction in predictions
            ]
        else:
            return [self.idx2class(i) for i in predictions]

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
        return classification_fscore(
            dataset.label, predictions, self.is_multilabel, classes=self.classes
        )

    def format_score(self, score_df: pd.DataFrame, format: str = "markdown") -> str:
        return tabulate(score_df, headers="keys", tablefmt="github", showindex=False)

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "is_multilabel": self.is_multilabel}

    @classmethod
    def _filter_config(cls, config):
        config = super()._filter_config(config)
        config.pop("algorithm", None)
        config.pop("task", None)
        return config

    def get_custom_objects(self) -> Dict[str, Any]:
        return {
            **super().get_custom_objects(),
            "PrecisionWithLogits": PrecisionWithLogits,
            "RecallWithLogits": RecallWithLogits,
            "FBetaScoreWithLogits": FBetaScoreWithLogits,
        }