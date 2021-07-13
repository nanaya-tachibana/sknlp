from __future__ import annotations
from typing import Sequence, Any, Optional

import pandas as pd
import tensorflow as tf

from sknlp.data import TaggingDataset
from sknlp.callbacks import TaggingFScoreMetric
from sknlp.layers import CrfDecodeLayer, CrfLossLayer, MLPLayer
from sknlp.utils.tagging import tagging_fscore, convert_ids_to_tags, Tag

from ..supervised_model import SupervisedNLPModel
from ..text2vec import Text2vec


class DeepTagger(SupervisedNLPModel):
    dataset_class = TaggingDataset
    dataset_args = ["use_crf", "add_start_end_tag"]

    def __init__(
        self,
        classes: Sequence[str],
        add_start_end_tag: bool = False,
        use_crf: bool = False,
        crf_learning_rate_multiplier: float = 1.0,
        max_sequence_length: Optional[int] = None,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 128,
        fc_activation: str = "tanh",
        text2vec: Optional[Text2vec] = None,
        **kwargs,
    ):
        self._add_start_end_tag = add_start_end_tag
        self._use_crf = use_crf
        self._crf_learning_rate_multiplier = crf_learning_rate_multiplier
        classes = list(classes)
        if use_crf and classes[0] != "O":
            classes.insert(0, "O")
        super().__init__(
            classes,
            max_sequence_length=max_sequence_length,
            text2vec=text2vec,
            task="tagging",
            **kwargs,
        )
        self.num_fc_layers = num_fc_layers
        self.fc_hidden_size = fc_hidden_size
        self.fc_activation = fc_activation

    @property
    def add_start_end_tag(self) -> bool:
        return self._add_start_end_tag

    @property
    def use_crf(self) -> bool:
        return self._use_crf

    @property
    def crf_learning_rate_multiplier(self) -> float:
        return self._crf_learning_rate_multiplier

    def get_loss(self) -> None:
        return None

    def get_callbacks(self, *args, **kwargs) -> list[tf.keras.callbacks.Callback]:
        callbacks = super().get_callbacks(*args, **kwargs)
        callbacks.append(TaggingFScoreMetric(self.classes))
        return callbacks

    def get_metrics(self) -> list[tf.keras.metrics.Metric]:
        return []

    @classmethod
    def get_monitor(cls) -> str:
        return "val_tag_accuracy"

    def build_output_layer(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        embeddings, mask, tag_ids = inputs
        emissions = MLPLayer(
            self.num_fc_layers,
            hidden_size=self.fc_hidden_size,
            output_size=self.num_classes * 2 + 1,
            activation=self.fc_activation,
            name="mlp",
        )(embeddings)
        return CrfLossLayer(
            self.num_classes * 2 + 1,
            max_sequence_length=self.max_sequence_length,
            learning_rate_multiplier=self.crf_learning_rate_multiplier,
        )([emissions, tag_ids], mask)

    def predict(
        self,
        X: Sequence[str] = None,
        *,
        dataset: TaggingDataset = None,
        thresholds: float = 0.5,
        batch_size: int = 128,
    ) -> list[list[Tag]]:
        tag_ids_list = super().predict(X, dataset=dataset, batch_size=batch_size)
        predictions = []
        for tag_ids in tag_ids_list:
            predictions.append(
                convert_ids_to_tags(
                    tag_ids.numpy().tolist(), self.idx2class, self.add_start_end_tag
                )
            )
        return predictions

    def score(
        self,
        X: Sequence[str] = None,
        y: Sequence[Sequence[str]] = None,
        *,
        dataset: TaggingDataset = None,
        thresholds: float = 0.5,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        dataset = self.prepare_dataset(X, y, dataset)
        predictions = self.predict(dataset=dataset, batch_size=batch_size)
        return tagging_fscore(
            [[Tag(*l) for l in yi] for yi in dataset.y], predictions, self.classes[1:]
        )

    def export(self, directory: str, name: str, version: str = "0") -> None:
        if self.use_crf:
            mask = self._model.get_layer("mask_layer").output
            emissions = self._model.get_layer("mlp").output
            crf = CrfDecodeLayer(
                self.num_classes * 2 + 1,
                learning_rate_multiplier=self.crf_learning_rate_multiplier,
                max_sequence_length=self.max_sequence_length,
            )
            crf.build(
                [tf.TensorShape([None, None, None]), tf.TensorShape([None, None])]
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
            "add_start_end_tag": self.add_start_end_tag,
            "use_crf": self.use_crf,
            "crf_learning_rate_multiplier": self._crf_learning_rate_multiplier,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DeepTagger":
        config.pop("algorithm", None)
        config.pop("task", None)
        return super().from_config(config)