from __future__ import annotations
from typing import Sequence, Any, Optional

import pandas as pd
import tensorflow as tf

from sknlp.data import TaggingDataset
from sknlp.callbacks import TaggingFScoreMetric
from sknlp.metrics import PrecisionWithLogits, RecallWithLogits, FBetaScoreWithLogits
from sknlp.layers import CrfDecodeLayer, CrfLossLayer, MLPLayer, GlobalPointerLayer
from sknlp.losses import MultiLabelCategoricalCrossentropy
from sknlp.utils.tagging import (
    Tag,
    tagging_fscore,
    convert_ids_to_tags,
    convert_global_pointer_to_tags,
)

from ..supervised_model import SupervisedNLPModel
from ..text2vec import Text2vec


class DeepTagger(SupervisedNLPModel):
    dataset_class = TaggingDataset
    dataset_args = ["output_format", "add_start_end_tag"]

    def __init__(
        self,
        classes: Sequence[str],
        add_start_end_tag: bool = False,
        output_format: str = "global_pointer",
        global_pointer_head_size: int = 64,
        crf_learning_rate_multiplier: float = 1.0,
        max_sequence_length: Optional[int] = None,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 256,
        fc_activation: str = "tanh",
        text2vec: Optional[Text2vec] = None,
        **kwargs,
    ):
        self._add_start_end_tag = add_start_end_tag
        if output_format not in {"bio", "global_pointer"}:
            raise ValueError(
                f"output_format必须为'bio'或者'global_pointer', 目前为{output_format}"
            )
        self._output_format = output_format
        self._crf_learning_rate_multiplier = crf_learning_rate_multiplier
        classes = list(classes)
        if output_format == "bio" and classes[0] != "O":
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
        self.global_pointer_head_size = global_pointer_head_size

    @property
    def add_start_end_tag(self) -> bool:
        return self._add_start_end_tag

    @property
    def output_format(self) -> str:
        return self._output_format

    @property
    def crf_learning_rate_multiplier(self) -> float:
        return self._crf_learning_rate_multiplier

    @property
    def num_bio_tags(self) -> int:
        return self.num_classes * 2 - 1

    def get_loss(self) -> None:
        if self.output_format == "bio":
            return None
        else:
            return MultiLabelCategoricalCrossentropy(flatten_axis=2)

    def get_callbacks(self, *args, **kwargs) -> list[tf.keras.callbacks.Callback]:
        callbacks = super().get_callbacks(*args, **kwargs)
        if self.output_format == "bio":
            callbacks.append(TaggingFScoreMetric(self.classes, self.add_start_end_tag))
        return callbacks

    def get_metrics(self) -> list[tf.keras.metrics.Metric]:
        if self.output_format == "bio":
            return []
        return [
            PrecisionWithLogits(activation="sigmoid"),
            RecallWithLogits(activation="sigmoid"),
            FBetaScoreWithLogits(self.num_classes, activation="sigmoid"),
        ]

    def get_monitor(self) -> str:
        if self.output_format == "bio":
            return "val_tag_accuracy"
        else:
            return "val_fbeta_score"

    def build_output_layer(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        if self.output_format == "bio":
            encodings, mask, tag_ids = inputs
            emissions = MLPLayer(
                self.num_fc_layers,
                hidden_size=self.fc_hidden_size,
                output_size=self.num_bio_tags,
                activation=self.fc_activation,
                name="mlp",
            )(encodings)
            crf_layer = CrfLossLayer(self.num_bio_tags)
            if self.crf_learning_rate_multiplier != 1:
                self._layerwise_learning_rate_multiplier.append(
                    (crf_layer, self.crf_learning_rate_multiplier)
                )
            return crf_layer([emissions, tag_ids], mask)
        else:
            encodings, mask = inputs
            return GlobalPointerLayer(
                self.num_classes,
                self.global_pointer_head_size,
                self.max_sequence_length + 2 * self.add_start_end_tag,
            )(encodings, mask)

    def predict(
        self,
        X: Sequence[str] = None,
        *,
        dataset: TaggingDataset = None,
        thresholds: float = 0.5,
        batch_size: int = 128,
    ) -> list[list[Tag]]:
        dataset = self.prepare_dataset(X, None, dataset, evaluation=True)
        raw_predictions = super().predict(X, dataset=dataset, batch_size=batch_size)
        tokens_list = dataset.tokenize(dataset.X)
        predictions: list[list[Tag]] = []
        if self.output_format == "bio":
            for tag_ids, tokens in zip(raw_predictions, tokens_list):
                itoken2ichar = self.vocab.create_itoken2ichar_mapping(tokens)
                predictions.append(
                    convert_ids_to_tags(
                        tag_ids.numpy().tolist(),
                        self.idx2class,
                        itoken2ichar,
                        self.add_start_end_tag,
                    )
                )
        else:
            for pointer, tokens in zip(raw_predictions, tokens_list):
                itoken2ichar = self.vocab.create_itoken2ichar_mapping(tokens)
                predictions.append(
                    convert_global_pointer_to_tags(
                        pointer.to_tensor(tf.float32.min).numpy(),
                        thresholds,
                        self.idx2class,
                        itoken2ichar,
                        self.add_start_end_tag,
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
        dataset = self.prepare_dataset(X, y, dataset, evaluation=True)
        predictions = self.predict(
            dataset=dataset, batch_size=batch_size, thresholds=thresholds
        )
        classes = self.classes
        if self.output_format == "bio":
            classes = classes[1:]
        return tagging_fscore(
            [[Tag(*l) for l in yi] for yi in dataset.y], predictions, classes
        )

    def build_inference_model(self) -> tf.keras.Model:
        if self.output_format == "bio":
            mask = self._model.get_layer("mask_layer").output
            emissions = self._model.get_layer("mlp").output
            crf = CrfDecodeLayer(self.num_bio_tags)
            crf.build(
                [tf.TensorShape([None, None, None]), tf.TensorShape([None, None])]
            )
            crf.set_weights(self._model.get_layer("crf").get_weights())
            model = tf.keras.Model(
                inputs=self._model.inputs[:-1], outputs=crf(emissions, mask)
            )
        else:
            model = self._model
        self._inference_model = model

    def export(self, directory: str, name: str, version: str = "0") -> None:
        original_model = self._inference_model
        if self.output_format == "bio":
            fill_value = tf.cast(0, tf.int32)
        else:
            fill_value = tf.float32.min
        self._inference_model = tf.keras.Model(
            inputs=self._inference_model.inputs,
            outputs=self._inference_model.outputs[0].to_tensor(fill_value),
        )
        super().export(directory, name, version=version)
        self._inference_model = original_model

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "add_start_end_tag": self.add_start_end_tag,
            "output_format": self.output_format,
            "crf_learning_rate_multiplier": self._crf_learning_rate_multiplier,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DeepTagger":
        config.pop("algorithm", None)
        config.pop("task", None)
        return super().from_config(config)