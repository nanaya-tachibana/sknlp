from __future__ import annotations
import itertools
from typing import Sequence, Any, Optional

import pandas as pd
from tabulate import tabulate
import tensorflow as tf
import numpy as np
from scipy.special import expit

from sknlp.data import TaggingDataset
from sknlp.callbacks import TaggingFScoreMetric
from sknlp.layers import CrfDecodeLayer, CrfLossLayer, MLPLayer
from sknlp.utils.tagging import tagging_fscore, convert_ids_to_tags, Tag

from ..supervised_model import SupervisedNLPModel
from ..text2vec import Text2vec


def _create_classes(
    classes: Sequence[str],
    use_crf: bool,
    pad_tag: str,
    start_tag: Optional[str] = None,
    end_tag: Optional[str] = None,
) -> list[str]:
    classes = list(classes)
    if not use_crf:
        return classes

    if "O" in classes:
        return classes
    classes = list(
        itertools.chain.from_iterable(
            [("-".join(["B", c]), "-".join(["I", c])) for c in classes]
        )
    )
    classes.insert(0, pad_tag)
    classes.append("O")
    if start_tag is not None and end_tag is not None:
        classes.append(start_tag)
        classes.append(end_tag)
    return classes


class DeepTagger(SupervisedNLPModel):
    def __init__(
        self,
        classes: Sequence[str],
        start_tag: Optional[str] = None,
        end_tag: Optional[str] = None,
        use_crf: bool = False,
        crf_learning_rate_multiplier: float = 1.0,
        max_sequence_length: Optional[int] = None,
        sequence_length: Optional[int] = None,
        segmenter: str = "char",
        embedding_size: int = 100,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 128,
        fc_activation: str = "tanh",
        text2vec: Optional[Text2vec] = None,
        **kwargs,
    ):
        self._start_tag = start_tag
        self._end_tag = end_tag
        self._pad_tag = "[PAD]"
        self._use_crf = use_crf
        self._crf_learning_rate_multiplier = crf_learning_rate_multiplier
        classes = _create_classes(
            classes, self.use_crf, self.pad_tag, self.start_tag, self.end_tag
        )
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
        self.num_fc_layers = num_fc_layers
        self.fc_hidden_size = fc_hidden_size
        self.fc_activation = fc_activation

    @property
    def use_crf(self) -> bool:
        return self._use_crf

    @property
    def crf_learning_rate_multiplier(self) -> float:
        return self._crf_learning_rate_multiplier

    @property
    def start_tag(self) -> Optional[str]:
        return self._start_tag

    @property
    def end_tag(self) -> Optional[str]:
        return self._end_tag

    @property
    def pad_tag(self) -> str:
        return self._pad_tag

    def get_loss(self) -> None:
        return None

    def get_callbacks(self, *args, **kwargs) -> list[tf.keras.callbacks.Callback]:
        callbacks = super().get_callbacks(*args, **kwargs)
        callbacks.append(
            TaggingFScoreMetric(
                self.idx2class,
                list(set([c.split("-")[-1] for c in self.classes if "-" in c])),
                self.pad_tag,
                start_tag=self.start_tag,
                end_tag=self.end_tag,
            )
        )
        return callbacks

    def get_metrics(self) -> list[tf.keras.metrics.Metric]:
        return []

    @classmethod
    def get_monitor(cls) -> str:
        return "val_fscore"

    def create_dataset_from_df(
        self,
        df: pd.DataFrame,
        no_label: bool = False,
    ) -> TaggingDataset:
        return TaggingDataset(
            self.text2vec.vocab,
            self.classes,
            df=df,
            max_length=self.max_sequence_length,
            no_label=no_label,
            text_segmenter=self.text2vec.segmenter,
            use_crf=self.use_crf,
            start_tag=self.start_tag,
            end_tag=self.end_tag,
        )

    def create_dataset_from_csv(
        self,
        filename: str,
        no_label: bool = False,
    ) -> TaggingDataset:
        return TaggingDataset(
            self.text2vec.vocab,
            self.classes,
            csv_file=filename,
            max_length=self.max_sequence_length,
            no_label=no_label,
            text_segmenter=self.text2vec.segmenter,
            use_crf=self.use_crf,
            start_tag=self.start_tag,
            end_tag=self.end_tag,
        )

    def build_output_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        embeddings, mask, tag_ids = inputs
        emissions = MLPLayer(
            self.num_fc_layers,
            hidden_size=self.fc_hidden_size,
            output_size=self.num_classes,
            activation=self.fc_activation,
            name="mlp",
        )(embeddings)
        return CrfLossLayer(
            self.num_classes,
            max_sequence_length=self.max_sequence_length,
            learning_rate_multiplier=self.crf_learning_rate_multiplier,
        )([emissions, tag_ids], mask)

    def predict(
        self,
        X: Sequence[str] = None,
        *,
        dataset: TaggingDataset = None,
        batch_size: int = 128,
    ) -> list[list[Tag]]:
        tag_ids_list = super().predict(X, dataset=dataset, batch_size=batch_size)
        predictions = []
        for tag_ids in tag_ids_list:
            predictions.append(
                convert_ids_to_tags(
                    self.idx2class,
                    tag_ids.numpy().tolist(),
                    self.pad_tag,
                    start_tag=self.start_tag,
                    end_tag=self.end_tag,
                )
            )
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
        return tagging_fscore(
            [[Tag(*l) for l in yi] for yi in dataset.y], predictions, labels
        )

    @classmethod
    def format_score(self, score_df: pd.DataFrame, format: str = "markdown") -> str:
        return tabulate(score_df, headers="keys", tablefmt="github", showindex=False)

    def export(self, directory: str, name: str, version: str = "0") -> None:
        if self.use_crf:
            mask = self._model.get_layer("mask_layer").output
            emissions = self._model.get_layer("mlp").output
            crf = CrfDecodeLayer(
                self.num_classes,
                learning_rate_multiplier=self.crf_learning_rate_multiplier,
                max_sequence_length=self.max_sequence_length,
            )
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

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "start_tag": self.start_tag,
            "end_tag": self.end_tag,
            "use_crf": self.use_crf,
            "crf_learning_rate_multiplier": self._crf_learning_rate_multiplier,
        }

    @classmethod
    def _filter_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        config = super()._filter_config(config)
        config.pop("algorithm", None)
        config.pop("task", None)
        return config

    @classmethod
    def get_custom_objects(cls) -> dict[str, Any]:
        return {
            **super().get_custom_objects(),
            "MLPLayer": MLPLayer,
            "CrfLossLayer": CrfLossLayer,
            "CrfDecodeLayer": CrfDecodeLayer,
            "Orthogonal": tf.keras.initializers.Orthogonal,
        }