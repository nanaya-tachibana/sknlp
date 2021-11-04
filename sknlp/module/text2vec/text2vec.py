from __future__ import annotations
from typing import Optional, Any, Union

import shutil
import tempfile

import tensorflow as tf

from sknlp.vocab import Vocab
from sknlp.utils import make_tarball

from sknlp.module.unsupervised_model import UnsupervisedNLPModel


class Text2vec(UnsupervisedNLPModel):
    def __init__(
        self,
        vocab: Vocab,
        segmenter: Optional[str] = None,
        max_sequence_length: Optional[int] = None,
        sequence_length: Optional[int] = None,
        embedding_size: int = 100,
        name: str = "text2vec",
        **kwargs,
    ) -> None:
        super().__init__(
            vocab,
            segmenter=segmenter,
            max_sequence_length=max_sequence_length,
            sequence_length=sequence_length,
            name=name,
            **kwargs,
        )
        self.embedding_size = embedding_size

    def __call__(
        self, inputs: Union[tf.Tensor, list[tf.Tensor]]
    ) -> Union[tf.Tensor, list[tf.Tensor]]:
        raise NotImplementedError()

    def update_dropout(self, dropout: float, **kwargs) -> None:
        pass

    def tokenize(self, text: str) -> list[int]:
        return self._tokenizer.tokenize(text)

    def get_inputs(self) -> Union[list[tf.Tensor], tf.Tensor]:
        return self.inputs

    @classmethod
    def load(cls, directory: str, epoch: Optional[int] = None) -> "Text2vec":
        module = super().load(directory, epoch=epoch)
        module.pretrain_layer = module._model.get_layer(module.name)
        return module

    def save_archive(self, filename: str) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.save(temp_dir)
            if not filename.endswith(".tar"):
                filename = ".".join([filename, "tar"])
            make_tarball(filename, temp_dir)

    @classmethod
    def load_archive(cls, filename: str) -> "Text2vec":
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(filename, temp_dir, format="tar")
            return cls.load(temp_dir)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "embedding_size": self.embedding_size,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Text2vec":
        config.pop("algorithm", None)
        return super().from_config(config)