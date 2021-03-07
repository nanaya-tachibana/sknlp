from typing import Sequence, List, Dict, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate
import tensorflow as tf
import tensorflow_addons as tfa

from sknlp.vocab import Vocab
from sknlp.data import SimilarityDataset
from sknlp.metrics import BinaryAccuracyWithLogits

from ..supervised_model import SupervisedNLPModel
from ..classifiers.utils import (
    classification_fscore,
    logits2probabilities,
    probabilities2classes,
)
from ..text2vec import Text2vec


class DeepDiscriminator(SupervisedNLPModel):
    def __init__(
        self,
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
        super().__init__(
            ["相似度"],
            max_sequence_length=max_sequence_length,
            sequence_length=sequence_length,
            segmenter=segmenter,
            embedding_size=embedding_size,
            text2vec=text2vec,
            task="similarity",
            **kwargs
        )
        self._use_batch_normalization = use_batch_normalization
        self._loss = loss
        self._loss_kwargs = loss_kwargs

    @property
    def use_batch_normalization(self):
        return self._use_batch_normalization

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
        return "binary_accuracy"

    def create_dataset_from_df(
        self, df: pd.DataFrame, vocab: Vocab, segmenter: str, labels: Sequence[str]
    ) -> SimilarityDataset:
        return SimilarityDataset(
            vocab,
            list(labels),
            df=df,
            max_length=self.max_sequence_length,
            text_segmenter=segmenter,
        )

    def dummy_y(self, X: Sequence[str]) -> List[str]:
        return ["O" for _ in range(len(X))]

    def predict_proba(
        self,
        X: Sequence[Tuple[str, str]] = None,
        *,
        dataset: SimilarityDataset = None,
        batch_size: int = 128
    ) -> np.ndarray:
        logits = super().predict(X=X, dataset=dataset, batch_size=batch_size)
        return logits2probabilities(logits, True)

    def predict(
        self,
        X: Sequence[Tuple[str, str]] = None,
        *,
        dataset: SimilarityDataset = None,
        batch_size: int = 128
    ) -> List[float]:
        return self.predict_proba(X=X, dataset=dataset, batch_size=batch_size)

    def score(
        self,
        X: Sequence[Tuple[str, str]] = None,
        y: Sequence[float] = None,
        *,
        dataset: SimilarityDataset = None,
        threshold: float = 0.5,
        batch_size: int = 128
    ) -> pd.DataFrame:
        dataset = self.prepare_dataset(X, y, dataset)
        probs = self.predict(dataset=dataset, batch_size=batch_size)
        return classification_fscore(
            dataset.label, probabilities2classes(probs, True), True, classes=[0, 1]
        )

    @classmethod
    def format_score(cls, score_df: pd.DataFrame, format: str = "markdown") -> str:
        return tabulate(score_df, headers="keys", tablefmt="github", showindex=False)

    def get_config(self) -> Dict[str, Any]:
        return {
            **super().get_config(),
            "use_batch_normalization": self.use_batch_normalization,
        }

    @classmethod
    def _filter_config(cls, config):
        config = super()._filter_config(config)
        config.pop("algorithm", None)
        config.pop("task", None)
        config.pop("classes", None)
        return config

    @classmethod
    def get_custom_objects(cls) -> Dict[str, Any]:
        return {
            **super().get_custom_objects(),
            "BinaryAccuracyWithLogits": BinaryAccuracyWithLogits,
        }