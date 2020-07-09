from typing import Sequence, Optional, Dict, Any, Callable, List

import json
import os
import itertools
from collections import Counter

import tensorflow as tf

from ..vocab import Vocab


class BaseNLPModel:

    def __init__(
        self,
        max_sequence_length: Optional[int] = None,
        sequence_length: Optional[int] = None,
        name: Optional[str] = None
    ) -> None:
        self._max_sequence_length = max_sequence_length
        self._sequence_length = sequence_length
        self._name = name
        self._built = False

    @staticmethod
    def build_vocab(
        texts: Sequence[str],
        segment_func: Callable[[str], Sequence[str]],
        min_frequency=5
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
        self._model = tf.keras.Model(
            inputs=self.get_inputs(), outputs=self.get_outputs(), name=self._name
        )
        self._built = True

    def get_inputs(self) -> tf.Tensor:
        raise NotImplementedError()

    @property
    def input_names(self) -> List[str]:
        return []

    @property
    def input_types(self) -> List[str]:
        return []

    @property
    def input_shapes(self) -> List[List[int]]:
        return []

    def get_outputs(self) -> tf.Tensor:
        raise NotImplementedError()

    @property
    def output_names(self) -> List[str]:
        return []

    @property
    def output_types(self) -> List[str]:
        return []

    @property
    def output_shapes(self) -> List[List[int]]:
        return []

    def get_loss(self):
        raise NotImplementedError()

    def get_metrics(self):
        raise NotImplementedError()

    def get_callbacks(self):
        raise NotImplementedError()

    def get_custom_objects(self):
        return {}

    def freeze(self) -> None:
        for layer in self._model.layers:
            layer.trainable = False

    def save_config(self, directory: str, filename: str = "meta.json") -> None:
        with open(os.path.join(directory, filename), "w") as f:
            f.write(json.dumps(self.get_config()))

    def save(self, directory: str) -> None:
        self._model.save(os.path.join(directory, "model"), save_format="tf")
        self.save_config(directory)

    @classmethod
    def load(cls, directory: str) -> "BaseNLPModel":
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.loads(f.read())
        module = cls.from_config(meta)
        module._model = tf.keras.models.load_model(
            os.path.join(directory, "model"),
            custom_objects=module.get_custom_objects()
        )
        module._built = True
        return module

    def export(self, directory: str, name: str, version: str = "0") -> None:
        d = os.path.join(directory, name, version)
        self._model.save(d, save_format="tf")
        self.save_config(d)

    def get_config(self) -> Dict[str, Any]:
        return {
            "inputs": self.input_names,
            "input_types": self.input_types,
            "input_shapes": self.input_shapes,
            "outputs": self.output_names,
            "output_types": self.output_types,
            "output_shapes": self.output_shapes,
            "max_sequence_length": self._max_sequence_length,
            "sequence_length": self._sequence_length,
            "name": self._name
        }

    @classmethod
    def _filter_config(cls, config):
        config.pop("inputs", None)
        config.pop("input_types", None)
        config.pop("input_shapes", None)
        config.pop("outputs", None)
        config.pop("output_types", None)
        config.pop("output_shapes", None)
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseNLPModel":
        return cls(**cls._filter_config(config))
