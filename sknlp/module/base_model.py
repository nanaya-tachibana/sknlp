from __future__ import annotations
from typing import Sequence, Optional, Any, Callable, Union

import json
import os
import itertools
from collections import Counter

import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import DisableSharedObjectScope

import sknlp
from sknlp.vocab import Vocab


class BaseNLPModel:
    def __init__(
        self,
        max_sequence_length: Optional[int] = None,
        sequence_length: Optional[int] = None,
        segmenter: Optional[str] = None,
        learning_rate_multiplier: Optional[dict[str, float]] = None,
        prediction_kwargs: Optional[dict[str, Any]] = None,
        custom_kwargs: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs
    ) -> None:
        self._max_sequence_length = max_sequence_length
        self._sequence_length = sequence_length
        self._segmenter = segmenter
        self._learning_rate_multiplier = learning_rate_multiplier or dict()
        self._layerwise_learning_rate_multiplier: list[
            tuple[tf.keras.layers.Layer, float]
        ] = []
        self._prediction_kwargs = prediction_kwargs or dict()
        self._custom_kwargs = custom_kwargs or dict()
        self._kwargs = kwargs

        self._name = name
        self._model: Optional[tf.keras.Model] = None
        self._inference_model: Optional[tf.keras.Model] = None
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
    def learning_rate_multiplier(self):
        for layer, multiplier in self._layerwise_learning_rate_multiplier:
            for variable in layer.variables:
                self._learning_rate_multiplier[variable.name] = multiplier
        return self._learning_rate_multiplier

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

    def build(self) -> None:
        if self._built:
            return
        self._model = tf.keras.Model(
            inputs=self.get_inputs(), outputs=self.get_outputs(), name=self.name
        )
        self._built = True

    def build_inference_model(self) -> tf.keras.Model:
        return self._model

    def build_preprocessing_layer(
        self, inputs: Union[tf.Tensor, list[tf.Tensor]]
    ) -> Union[tf.Tensor, list[tf.Tensor]]:
        return inputs

    def build_encoding_layer(
        self, inputs: Union[tf.Tensor, list[tf.Tensor]]
    ) -> Union[tf.Tensor, list[tf.Tensor]]:
        raise NotImplementedError()

    def build_intermediate_layer(
        self, inputs: Union[tf.Tensor, list[tf.Tensor]]
    ) -> Union[tf.Tensor, list[tf.Tensor]]:
        return inputs

    def build_output_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def get_inputs(self) -> tf.Tensor:
        raise NotImplementedError()

    def get_outputs(self) -> tf.Tensor:
        return self.build_output_layer(
            self.build_intermediate_layer(
                self.build_encoding_layer(
                    self.build_preprocessing_layer(self.get_inputs())
                )
            )
        )

    def get_loss(self, *args, **kwargs) -> Optional[tf.keras.losses.Loss]:
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
        with DisableSharedObjectScope():
            module._model = tf.keras.models.load_model(
                os.path.join(directory, cls._get_model_filename(epoch=epoch))
            )
            module._built = True
        return module

    def export(self, directory: str, name: str, version: str = "0") -> None:
        d = os.path.join(directory, name, version)

        model: tf.keras.Model = tf.keras.models.model_from_json(
            self._inference_model.to_json(),
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
            "__version__": sknlp.__version__,
            "__package__": sknlp.__name__,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseNLPModel":
        return cls(**config)