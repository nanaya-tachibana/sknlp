from __future__ import annotations
from typing import Optional, Any, Union

import json
import os
import shutil
import tempfile

import tensorflow as tf

from sknlp.vocab import Vocab
from sknlp.utils import make_tarball

from sknlp.module.base_model import BaseNLPModel
from .tokenizer import get_tokenizer


class Text2vec(BaseNLPModel):
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
        """
        基础符号->向量模块.

        Parameters:
        ----------
        vocab: sknlp.vocab.Vocab. 符号表.
        embed_size: int > 0. 向量维度.
        name: str. 模块名.
        embeddings_initializer: Initializer for the `embeddings` matrix.
        embeddings_regularizer: Regularizer function applied to
        the `embeddings` matrix.
        embeddings_constraint: Constraint function applied to
        the `embeddings` matrix.
        input_length: Length of input sequences, when it is constant.
        This argument is required if you are going to connect
        `Flatten` then `Dense` layers upstream
        (without it, the shape of the dense outputs cannot be computed).

        Inputs
        ----------
        2D tensor with shape: `(batch_size, input_length)`.

        Outputs
        ----------
        3D tensor with shape: `(batch_size, input_length, embed_size)`.
        """
        super().__init__(
            segmenter=segmenter,
            max_sequence_length=max_sequence_length,
            sequence_length=sequence_length,
            name=name,
            **kwargs,
        )
        self.embedding_size = embedding_size
        self._vocab = vocab
        self._model: tf.keras.Model = None
        self._tokenizer = get_tokenizer(segmenter, self.vocab)

    def __call__(
        self, inputs: Union[tf.Tensor, list[tf.Tensor]]
    ) -> Union[tf.Tensor, list[tf.Tensor]]:
        return self._model(inputs)

    def compute_mask(
        self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        raise NotImplementedError()

    def update_dropout(self, dropout: float) -> None:
        pass

    @property
    def vocab(self) -> Vocab:
        return self._vocab

    @property
    def segmenter(self) -> str:
        return self._segmenter

    def tokenize(self, text: str) -> list[int]:
        return self._tokenizer.tokenize(text)

    @property
    def weights(self) -> list[tf.Tensor]:
        return self._model.get_weights()

    @weights.setter
    def weights(self, weights: list[tf.Tensor]) -> None:
        self._model.set_weights(weights)

    def save_vocab(self, directory: str, filename: str = "vocab.json") -> None:
        with open(os.path.join(directory, filename), "w", encoding="UTF-8") as f:
            f.write(self.vocab.to_json())

    def save(self, directory: str) -> None:
        super().save(directory)
        self.save_vocab(directory)

    @classmethod
    def load(cls, directory: str) -> "Text2vec":
        with open(os.path.join(directory, "meta.json"), encoding="UTF-8") as f:
            meta = json.loads(f.read())
        with open(os.path.join(directory, "vocab.json"), encoding="UTF-8") as f:
            vocab = Vocab.from_json(f.read())
        module = cls.from_config({"vocab": vocab, **meta})
        module._model = tf.keras.models.load_model(os.path.join(directory, "model"))
        module._built = True
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

    def export(self, directory: str, name: str, version: str = "0") -> None:
        super().export(directory, name, version)
        d = os.path.join(directory, name, version)
        self.save_vocab(d)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "embedding_size": self.embedding_size,
        }

    def get_inputs(self) -> tf.Tensor:
        return self._model.input

    def get_outputs(self) -> tf.Tensor:
        return self._model.output