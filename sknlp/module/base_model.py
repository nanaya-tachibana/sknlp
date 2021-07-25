from __future__ import annotations
from typing import Sequence, Optional, Any, Callable

import json
import os
import itertools
from collections import Counter

import tensorflow as tf

from sknlp.vocab import Vocab


class BaseNLPModel:
    def __init__(
        self,
        max_sequence_length: Optional[int] = None,
        sequence_length: Optional[int] = None,
        segmenter: Optional[str] = None,
        name: Optional[str] = None,
        prediction_kwargs: Optional[dict[str, Any]] = None,
        custom_kwargs: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> None:
        self._max_sequence_length = max_sequence_length
        self._sequence_length = sequence_length
        self._segmenter = segmenter
        self._name = name
        self._prediction_kwargs = prediction_kwargs or dict()
        self._custom_kwargs = custom_kwargs or dict()
        self._kwargs = kwargs
        self._model: tf.keras.Model = None
        self._built = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_sequence_length(self) -> Optional[int]:
        return self._max_sequence_length

    @property
    def sequence_length(self) -> Optional[int]:
        return self._sequence_length

    @property
    def segmenter(self):
        return self._segmenter

    @property
    def prediction_kwargs(self) -> dict[str, Any]:
        return self._prediction_kwargs

    @property
    def custom_kwargs(self) -> dict[str, Any]:
        return self._custom_kwargs

    @staticmethod
    def build_vocab(
        texts: Sequence[str],
        segment_func: Callable[[str], Sequence[str]],
        min_frequency=5,
    ) -> Vocab:
        counter = Counter(
            itertools.chain.from_iterable(segment_func(text) for text in texts)
        )
        return Vocab(counter, min_frequency=min_frequency)

    def build_encode_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def build_output_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def build(self) -> None:
        if self._built:
            return
        self._model = tf.keras.Model(
            inputs=self.get_inputs(), outputs=self.get_outputs(), name=self._name
        )
        self._built = True

    def get_inputs(self) -> tf.Tensor:
        raise NotImplementedError()

    def get_outputs(self) -> tf.Tensor:
        raise NotImplementedError()

    def get_loss(self, *args, **kwargs) -> tf.keras.losses.Loss:
        raise NotImplementedError()

    def get_metrics(self, *args, **kwargs) -> list[tf.keras.metrics.Metric]:
        return []

    def get_callbacks(self, *args, **kwargs) -> list[tf.keras.callbacks.Callback]:
        return []

    def get_monitor(self) -> str:
        raise NotImplementedError()

    @classmethod
    def _get_model_filename_template(cls) -> str:
        return "model_{epoch:04d}"

    @classmethod
    def _get_model_filename(cls, epoch: Optional[int] = None) -> str:
        if epoch is not None:
            if epoch < 1:
                epoch = 0
            return cls._get_model_filename_template().format(epoch=epoch)
        return "model"

    def freeze(self) -> None:
        for layer in self._model.layers:
            layer.trainable = False

    def save_config(self, directory: str, filename: str = "meta.json") -> None:
        with open(os.path.join(directory, filename), "w", encoding="UTF-8") as f:
            f.write(json.dumps(self.get_config(), ensure_ascii=False))

    def save(self, directory: str) -> None:
        self._model.save(
            os.path.join(directory, self._get_model_filename()), save_format="tf"
        )
        self.save_config(directory)

    @classmethod
    def load(cls, directory: str, epoch: Optional[int] = None) -> "BaseNLPModel":
        with open(os.path.join(directory, "meta.json"), encoding="UTF-8") as f:
            meta = json.loads(f.read())
        module = cls.from_config(meta)
        module._model = tf.keras.models.load_model(
            os.path.join(directory, cls._get_model_filename(epoch=epoch))
        )
        module._built = True
        return module

    def export(self, directory: str, name: str, version: str = "0") -> None:
        d = os.path.join(directory, name, version)

        model: tf.keras.Model = tf.keras.models.model_from_json(
            self._model.to_json(),
            custom_objects={
                "TruncatedNormal": tf.keras.initializers.TruncatedNormal,
                "GlorotUniform": tf.keras.initializers.GlorotUniform,
                "Orthogonal": tf.keras.initializers.Orthogonal,
                "Zeros": tf.keras.initializers.Zeros,
            },
        )
        model.set_weights(self._model.get_weights())
        model.save(d, include_optimizer=False, save_format="tf")
        self.save_config(d)

    def get_config(self) -> dict[str, Any]:
        return {
            "max_sequence_length": self.max_sequence_length,
            "sequence_length": self.sequence_length,
            "segmenter": self.segmenter,
            "name": self.name,
            "prediction_kwargs": self.prediction_kwargs,
            "custom_kwargs": self.custom_kwargs,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseNLPModel":
        return cls(**config)