from typing import Sequence, List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from tabulate import tabulate
import tensorflow as tf

from sknlp.vocab import Vocab
from sknlp.data import ClassificationDataset
from sknlp.metrics import PrecisionWithLogits, RecallWithLogits
from sknlp.callbacks import FScore

from ..supervised_model import SupervisedNLPModel
from ..text2vec import Text2vec

from .utils import (
    logits2probabilities,
    probabilities2classes,
    classification_fscore
)


class DeepClassifier(SupervisedNLPModel):

    def __init__(
        self,
        classes: Sequence[str],
        is_multilabel: bool = True,
        segmenter: str = "jieba",
        embedding_size: int = 100,
        max_sequence_length: int = 100,
        text2vec: Optional[Text2vec] = None,
        **kwargs
    ):
        super().__init__(classes,
                         segmenter=segmenter,
                         embedding_size=embedding_size,
                         max_sequence_length=max_sequence_length,
                         text2vec=text2vec,
                         task="classification",
                         **kwargs)
        self._is_multilabel = is_multilabel

    def get_loss(self):
        if self._is_multilabel:
            return tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            return tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        return [FScore()]

    def get_metrics(self) -> List[tf.keras.metrics.Metric]:
        if self._is_multilabel:
            return [PrecisionWithLogits(), RecallWithLogits()]
        else:
            return [PrecisionWithLogits(logits2scores="softmax"),
                    RecallWithLogits(logits2scores="softmax")]

    def create_dataset_from_df(
        self,
        df: pd.DataFrame,
        vocab: Vocab,
        segmenter: str,
        labels: Sequence[str]
    ) -> ClassificationDataset:
        return ClassificationDataset(
            vocab,
            list(labels),
            df=df,
            is_multilabel=self._is_multilabel,
            max_length=self._max_sequence_length,
            text_segmenter=segmenter
        )

    def predict_proba(
        self,
        X: Sequence[str] = None,
        *,
        dataset: ClassificationDataset = None,
        batch_size: int = 128
    ) -> np.ndarray:
        logits = super().predict(X=X, dataset=dataset, batch_size=batch_size)
        return logits2probabilities(logits, self._is_multilabel)

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
            probabilities, self._is_multilabel, thresholds=thresholds
        )
        if self._is_multilabel:
            return [
                [self._idx2class[i] for i in prediction] for prediction in predictions
            ]
        else:
            return [self._idx2class[i] for i in predictions]

    def score(
        self,
        X: Sequence[str] = None,
        y: Union[Sequence[Sequence[str]], Sequence[str]] = None,
        *,
        dataset: ClassificationDataset = None,
        thresholds: Union[float, List[float]] = 0.5,
        batch_size: int = 128
    ) -> pd.DataFrame:
        predictions = self.predict(
            X=X, dataset=dataset, thresholds=thresholds, batch_size=batch_size
        )
        return classification_fscore(
            y, predictions, self._is_multilabel, classes=list(self._class2idx.keys())
        )

    def format_score(self, score_df: pd.DataFrame, format: str = "markdown") -> str:
        return tabulate(score_df, headers="keys", tablefmt="github", showindex=False)

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "is_multilabel": self._is_multilabel}

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
            "RecallWithLogits": RecallWithLogits
        }
