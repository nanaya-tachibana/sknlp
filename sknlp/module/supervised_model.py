from __future__ import annotations
from typing import Optional, Any, Union

import tensorflow as tf

from sknlp.vocab import Vocab
from .text2vec import Text2vec
from .base_model import BaseNLPModel


class SupervisedNLPModel(BaseNLPModel):
    def __init__(
        self,
        classes: list[str],
        text2vec: Optional[Text2vec] = None,
        vocab: Optional[Vocab] = None,
        task: Optional[str] = None,
        algorithm: Optional[str] = None,
        **kwargs
    ) -> None:
        if text2vec is None and vocab is None:
            raise ValueError("`text2vec`和`vocab`不能都为None")
        super().__init__(vocab=vocab or text2vec.vocab, **kwargs)
        self._text2vec: Optional[Text2vec] = None
        self._text2vec_name: str = None
        if text2vec is not None:
            self.text2vec = text2vec

        self._task = task
        self._algorithm = algorithm
        self._class2idx = dict(zip(classes, range(len(classes))))
        self._idx2class = dict(zip(range(len(classes)), classes))

    @property
    def num_classes(self) -> int:
        return len(self._class2idx)

    @property
    def classes(self) -> list[str]:
        return list(self._class2idx.keys())

    @property
    def text2vec(self) -> Text2vec:
        return self._text2vec

    @text2vec.setter
    def text2vec(self, tv: Text2vec) -> None:
        if self.max_sequence_length is not None and tv.max_sequence_length is not None:
            self._max_sequence_length = min(
                self.max_sequence_length, tv.max_sequence_length
            )
        else:
            self._max_sequence_length = (
                self.max_sequence_length or tv.max_sequence_length
            )
        self._sequence_length = tv.sequence_length
        self._segmenter = tv.segmenter
        self._text2vec = tv
        self._text2vec_name = tv.name

    def get_inputs(self) -> Union[list[tf.Tensor], tf.Tensor]:
        if getattr(self, "inputs", None) is None:
            return self.text2vec.get_inputs()
        else:
            return self.inputs

    def class2idx(self, class_name: str) -> Optional[int]:
        return self._class2idx.get(class_name, None)

    def idx2class(self, class_idx: int) -> Optional[str]:
        return self._idx2class.get(class_idx, None)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "classes": self.classes,
            "task": self._task,
            "algorithm": self._algorithm,
            "text2vec": {"name": self._text2vec_name},
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SupervisedNLPModel":
        text2vec_dict = config.pop("text2vec", dict())
        module = super().from_config(config)
        module._text2vec_name = text2vec_dict.get("name", None)
        return module

    @classmethod
    def load(cls, directory: str, epoch: Optional[int] = None) -> "SupervisedNLPModel":
        module = super().load(directory, epoch=epoch)
        module.text2vec = Text2vec(
            module.vocab,
            segmenter=module.segmenter,
            max_sequence_length=module.max_sequence_length,
            sequence_length=module.sequence_length,
            name=module._text2vec_name,
        )
        module.build_inference_model()
        return module