from __future__ import annotations
from typing import Any, Sequence, Optional, Callable

import tensorflow as tf
import numpy as np

from sknlp.vocab import Vocab
from .nlp_dataset import NLPDataset
from .bert_mixin import BertDatasetMixin


def _generate_y(x):
    return tf.reshape(x, (-1, tf.shape(x)[-1])), tf.range(tf.shape(x)[0])


class RetrievalDataset(NLPDataset):
    def __init__(
        self,
        vocab: Vocab,
        labels: Sequence[str],
        segmenter: Optional[str] = None,
        X: Optional[Sequence[Any]] = None,
        y: Optional[Sequence[Any]] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        has_label: bool = True,
        max_length: Optional[int] = None,
        text_dtype: tf.DType = tf.int64,
        label_dtype: tf.DType = tf.int64,
        **kwargs,
    ):
        super().__init__(
            vocab,
            segmenter,
            X=X,
            y=y,
            csv_file=csv_file,
            in_memory=in_memory,
            has_label=has_label,
            max_length=max_length,
            na_value="",
            column_dtypes=["str", "str", "str"],
            text_dtype=text_dtype,
            label_dtype=label_dtype,
        )

    @property
    def y(self) -> list[float]:
        return []

    @property
    def batch_padding_shapes(self) -> tuple:
        return (None, None)

    def _format_y(self, y: Sequence[str | Sequence[str]]) -> list[Sequence[str]]:
        if isinstance(y[0], str):
            y = [y]
        return y

    def py_transform(self, *data: Sequence[tf.Tensor]) -> list[Any]:
        token_ids_list = self.vocab.token2idx(self.py_text_transform(data))
        return tf.keras.preprocessing.sequence.pad_sequences(
            token_ids_list, padding="post"
        )

    def tf_transform_after_py_transform(self, data: tf.Tensor) -> tf.Tensor:
        if self.has_label:
            return data
        else:
            return tf.concat([data, data], 0)

    def py_transform_out_dtype(self) -> tf.DType:
        return self.text_dtype

    def batchify(
        self,
        batch_size: int,
        shuffle: bool = True,
        training: bool = True,
        shuffle_buffer_size: Optional[int] = None,
        after_batch: Optional[Callable] = None,
    ) -> tf.data.Dataset:
        if after_batch is None:
            after_batch = _generate_y
        return super().batchify(
            batch_size,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            after_batch=after_batch,
        )


def _bert_generate_y(x):
    reshaped_x = tf.reshape(x, (-1, tf.shape(x)[-1]))
    return (
        (reshaped_x, tf.zeros_like(reshaped_x, dtype=tf.int64)),
        tf.range(tf.shape(x)[0]),
    )


class BertRetrievalDataset(BertDatasetMixin, RetrievalDataset):
    @property
    def batch_padding_shapes(self) -> tuple:
        return (None, None)

    def tf_transform_before_py_transform(self, *data: Sequence[tf.Tensor]) -> tf.Tensor:
        return self.tokenize(data)

    def py_transform_out_dtype(self) -> tf.DType:
        return self.text_dtype

    def py_text_transform(self, tokens_tensor: tf.Tensor) -> list[np.ndarray]:
        cls_id: int = self.vocab["[CLS]"]
        sep_id: int = self.vocab["[SEP]"]
        pad_id: int = self.vocab[self.vocab.pad]
        token_ids_list = []
        for _, tokens in enumerate(tokens_tensor.numpy().tolist()):
            token_ids = self.vocab.token2idx(
                [token.decode("UTF-8") for token in tokens][: self.max_length]
            )
            token_ids = [tid for tid in token_ids if tid != pad_id]
            token_ids.insert(0, cls_id)
            token_ids.append(sep_id)
            token_ids_list.append(token_ids)
        return tf.keras.preprocessing.sequence.pad_sequences(
            token_ids_list, padding="post"
        )

    def py_transform(self, data: tf.Tensor) -> list[Any]:
        return self.py_text_transform(data)

    def batchify(
        self,
        batch_size: int,
        shuffle: bool = True,
        training: bool = True,
        shuffle_buffer_size: Optional[int] = None,
        after_batch: Optional[Callable] = None,
    ) -> tf.data.Dataset:
        if after_batch is None:
            after_batch = _bert_generate_y
        return super().batchify(
            batch_size,
            shuffle=shuffle,
            training=training,
            shuffle_buffer_size=shuffle_buffer_size,
            after_batch=after_batch,
        )