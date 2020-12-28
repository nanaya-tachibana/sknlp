from typing import Sequence, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from tabulate import tabulate
import tensorflow as tf

from sknlp.vocab import Vocab
from sknlp.data import TaggingDataset

from ..supervised_model import SupervisedNLPModel
from ..text2vec import Text2vec

from .utils import tagging_fscore


class DeepTagger(SupervisedNLPModel):
    def __init__(
        self,
        classes: Sequence[str],
        start_tag: Optional[str] = None,
        end_tag: Optional[str] = None,
        max_sequence_length: Optional[int] = None,
        sequence_length: Optional[int] = None,
        segmenter: str = "char",
        embedding_size: int = 100,
        text2vec: Optional[Text2vec] = None,
        loss: str = None,
        **kwargs
    ):
        classes = list(classes)
        if start_tag is not None and end_tag is not None:
            self._start_tag = start_tag
            self._end_tag = end_tag
            if start_tag not in classes:
                classes.append(start_tag)
            if end_tag not in classes:
                classes.append(end_tag)
        self._pad_tag = "[PAD]"
        if self._pad_tag != classes[0]:
            classes.insert(0, self._pad_tag)
        super().__init__(
            classes,
            max_sequence_length=max_sequence_length,
            sequence_length=sequence_length,
            segmenter=segmenter,
            embedding_size=embedding_size,
            text2vec=text2vec,
            task="tagging",
            **kwargs
        )
        self._loss = loss

    def get_loss(self) -> None:
        return None

    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        return []

    def get_metrics(self) -> List[tf.keras.metrics.Metric]:
        return []

    def get_monitor(self) -> str:
        return None

    def create_dataset_from_df(
        self, df: pd.DataFrame, vocab: Vocab, segmenter: str, labels: Sequence[str]
    ) -> TaggingDataset:
        return TaggingDataset(
            vocab,
            list(labels),
            df=df,
            max_length=self._max_sequence_length,
            text_segmenter=segmenter,
            start_tag=self._start_tag,
            end_tag=self._end_tag,
        )

    def dummy_y(self, X: Sequence[str]) -> List[List[str]]:
        return [["O" for _ in range(len(xi))] for xi in X]

    def predict(
        self,
        X: Sequence[str] = None,
        *,
        dataset: TaggingDataset = None,
        batch_size: int = 128
    ) -> List[List[str]]:
        tag_ids_list = super().predict(X, dataset=dataset, batch_size=batch_size)
        exclude_tags = {self._pad_tag, self._start_tag, self._end_tag}
        predictions = []
        for tag_ids in tag_ids_list:
            tags = []
            for tag_id in tag_ids:
                tag = self._idx2class[tag_id]
                if tag not in exclude_tags:
                    tags.append(tag)
                if tag == self._pad_tag:
                    break
            predictions.append(tags)
        return predictions

    def score(
        self,
        X: Sequence[str] = None,
        y: Sequence[Sequence[str]] = None,
        *,
        dataset: TaggingDataset = None,
        batch_size: int = 128
    ) -> pd.DataFrame:
        predictions = self.predict(X=X, dataset=dataset, batch_size=batch_size)
        labels = [c.split("-")[-1] for c in self.classes if "-" in c]
        return tagging_fscore(X, y, predictions, labels)

    def format_score(self, score_df: pd.DataFrame, format: str = "markdown") -> str:
        return tabulate(score_df, headers="keys", tablefmt="github", showindex=False)

    def get_config(self) -> Dict[str, Any]:
        return {
            **super().get_config(),
            "start_tag": self._start_tag,
            "end_tag": self._end_tag,
        }

    @classmethod
    def _filter_config(cls, config):
        config = super()._filter_config(config)
        config.pop("algorithm", None)
        config.pop("task", None)
        return config

    def get_custom_objects(self) -> Dict[str, Any]:
        return {
            **super().get_custom_objects(),
        }