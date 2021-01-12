from typing import Sequence, List, Dict, Any, Optional
from functools import partial

import pandas as pd
from tabulate import tabulate
import tensorflow as tf

from sknlp.vocab import Vocab
from sknlp.data import TaggingDataset
from sknlp.callbacks import ModelScoreCallback
from sknlp.layers import CrfDecodeLayer

from ..supervised_model import SupervisedNLPModel
from ..text2vec import Text2vec

from .utils import tagging_fscore, viterbi_decode


class DeepTagger(SupervisedNLPModel):
    def __init__(
        self,
        classes: Sequence[str],
        start_tag: Optional[str] = None,
        end_tag: Optional[str] = None,
        use_crf: bool = True,
        max_sequence_length: Optional[int] = None,
        sequence_length: Optional[int] = None,
        segmenter: str = "char",
        embedding_size: int = 100,
        text2vec: Optional[Text2vec] = None,
        **kwargs,
    ):
        classes = list(classes)
        self._start_tag = start_tag
        self._end_tag = end_tag
        if start_tag is not None and end_tag is not None:
            if start_tag not in classes:
                classes.append(start_tag)
            if end_tag not in classes:
                classes.append(end_tag)
        self._pad_tag = "[PAD]"
        if self._pad_tag != classes[0]:
            classes.insert(0, self._pad_tag)
        self._use_crf = use_crf
        super().__init__(
            classes,
            max_sequence_length=max_sequence_length,
            sequence_length=sequence_length,
            segmenter=segmenter,
            embedding_size=embedding_size,
            text2vec=text2vec,
            task="tagging",
            **kwargs,
        )

    @property
    def use_crf(self) -> bool:
        return self._use_crf

    @property
    def start_tag(self) -> str:
        return self._start_tag

    @property
    def end_tag(self) -> str:
        return self._end_tag

    @property
    def pad_tag(self) -> str:
        return self._pad_tag

    def get_loss(self) -> None:
        return None

    def get_callbacks(self, batch_size: int) -> List[tf.keras.callbacks.Callback]:
        callbacks = []
        if self.valid_dataset is not None:
            score_func = partial(
                self.score, dataset=self.valid_dataset, batch_size=batch_size
            )
            callbacks.append(ModelScoreCallback(score_func))
        return callbacks

    def get_metrics(self) -> List[tf.keras.metrics.Metric]:
        return []

    def get_monitor(self) -> str:
        return "val_fscore"

    def create_dataset_from_df(
        self, df: pd.DataFrame, vocab: Vocab, segmenter: str, labels: Sequence[str]
    ) -> TaggingDataset:
        return TaggingDataset(
            vocab,
            list(labels),
            df=df,
            max_length=self.max_sequence_length,
            text_segmenter=segmenter,
            start_tag=self.start_tag,
            end_tag=self.end_tag,
        )

    def dummy_y(self, X: Sequence[str]) -> List[List[str]]:
        return [["O" for _ in range(len(xi))] for xi in X]

    def predict(
        self,
        X: Sequence[str] = None,
        *,
        dataset: TaggingDataset = None,
        batch_size: int = 128,
    ) -> List[List[str]]:
        emissions, mask = super().predict(X, dataset=dataset, batch_size=batch_size)
        tag_ids_list = viterbi_decode(
            self._model.get_layer("crf").get_weights()[0],
            emissions.transpose(1, 0, 2),
            mask.transpose(1, 0),
        )
        exclude_tags = {self.pad_tag, self.start_tag, self.end_tag}
        predictions = []
        for tag_ids in tag_ids_list:
            tags = []
            for tag_id in tag_ids:
                tag = self.idx2class(tag_id)
                if tag not in exclude_tags:
                    tags.append(tag)
                if tag == self.pad_tag:
                    break
            predictions.append(tags)
        return predictions

    def score(
        self,
        X: Sequence[str] = None,
        y: Sequence[Sequence[str]] = None,
        *,
        dataset: TaggingDataset = None,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        dataset = self.prepare_dataset(X, y, dataset)
        predictions = self.predict(dataset=dataset, batch_size=batch_size)
        labels = list(set([c.split("-")[-1] for c in self.classes if "-" in c]))
        return tagging_fscore(dataset.text, dataset.label, predictions, labels)

    def format_score(self, score_df: pd.DataFrame, format: str = "markdown") -> str:
        return tabulate(score_df, headers="keys", tablefmt="github", showindex=False)

    def export(self, directory: str, name: str, version: str = "0") -> None:
        if self.use_crf:
            mask = self._model.get_layer("mask_layer").output
            emissions = self._model.get_layer("mlp").output
            crf = CrfDecodeLayer(self.num_classes, self.max_sequence_length)
            crf.set_weights(self._model.get_layer("crf").get_weights())
            model = tf.keras.Model(
                inputs=self._model.inputs[0], outputs=crf(emissions, mask)
            )
            original_model = self._model
            self._model = model
            super().export(directory, name, version=version)
            self._model = original_model
        else:
            super().export(directory, name, version=version)

    def get_config(self) -> Dict[str, Any]:
        return {
            **super().get_config(),
            "start_tag": self.start_tag,
            "end_tag": self.end_tag,
            "use_crf": self.use_crf
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
            "CrfDecodeLayer": CrfDecodeLayer,
        }