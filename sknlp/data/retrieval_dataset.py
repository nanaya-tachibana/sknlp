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
        labels: Sequence[int],
        segmenter: Optional[str] = None,
        X: Optional[Sequence[Any]] = None,
        y: Optional[Sequence[Any]] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        has_label: bool = True,
        max_length: Optional[int] = None,
        text_dtype: tf.DType = tf.int64,
        label_dtype: tf.DType = tf.int64,
        column_dtypes=["str", "str", "str"],
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
            column_dtypes=column_dtypes,
            text_dtype=text_dtype,
            label_dtype=label_dtype,
        )

    @property
    def X(self) -> list[Any]:
        return [
            data[0].decode("UTF-8")
            for data in self._original_dataset.as_numpy_iterator()
        ]

    @property
    def y(self) -> list[str]:
        if not self.has_label:
            return []
        return [
            [d.decode("UTF-8") for d in data[1:]]
            for data in self._original_dataset.as_numpy_iterator()
        ]

    @property
    def batch_padding_shapes(self) -> tuple:
        return (None, None)

    def normalize(self, *data: list[tf.Tensor]) -> list[tf.Tensor]:
        return self.normalize_text(*data)

    def _format_y(self, y: Sequence[str | Sequence[str]]) -> list[Sequence[str]]:
        return self._format_X(y)

    def py_transform(self, *data: Sequence[tf.Tensor]) -> np.ndarray:
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


def _reshape_x(*data):
    x = data[0]
    reshaped_x = tf.reshape(x, (-1, tf.shape(x)[-1]))
    if len(data) == 1:
        return [reshaped_x]
    else:
        return [reshaped_x, data[-1]]


class RetrievalEvaluationDataset(RetrievalDataset):
    def __init__(
        self,
        vocab: Vocab,
        labels: Sequence[int],
        segmenter: Optional[str] = None,
        X: Optional[Sequence[Any]] = None,
        y: Optional[Sequence[Any]] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        has_label: bool = True,
        max_length: Optional[int] = None,
        text_dtype: tf.DType = tf.int64,
        label_dtype: tf.DType = tf.int64,
        column_dtypes=["str", "str", "int"],
        **kwargs,
    ):
        super().__init__(
            vocab,
            labels,
            segmenter=segmenter,
            X=X,
            y=y,
            csv_file=csv_file,
            in_memory=in_memory,
            has_label=has_label,
            max_length=max_length,
            text_dtype=text_dtype,
            label_dtype=label_dtype,
            column_dtypes=column_dtypes,
            **kwargs,
        )

    @property
    def y(self) -> list[int]:
        if not self.has_label:
            return []
        return [int(data[-1]) for data in self._original_dataset.as_numpy_iterator()]

    @property
    def batch_padding_shapes(self) -> tuple:
        return [(None, None), ()][: None if self.has_label else -1]

    def normalize(self, *data: list[tf.Tensor]) -> list[tf.Tensor]:
        return super(RetrievalDataset, self).normalize(*data)

    def _format_y(self, y: Sequence[int]) -> list[Sequence[int]]:
        return super(RetrievalDataset, self)._format_y(y)

    def py_label_transform(self, label: tf.Tensor) -> tf.Tensor:
        return label

    def py_transform(self, *data: Sequence[tf.Tensor]) -> list[Any]:
        token_ids_list = self.vocab.token2idx(
            self.py_text_transform(data[: -1 if self.has_label else None])
        )
        transformed_data = [
            tf.keras.preprocessing.sequence.pad_sequences(
                token_ids_list, padding="post"
            )
        ]
        if self.has_label:
            transformed_data.append(self.py_label_transform(data[-1]))
        return transformed_data

    def tf_transform_after_py_transform(
        self, *data: Sequence[tf.Tensor]
    ) -> Sequence[tf.Tensor]:
        return data

    def py_transform_out_dtype(self) -> tf.DType:
        return super(RetrievalDataset, self).py_transform_out_dtype()

    def batchify(
        self,
        batch_size: int,
        shuffle: bool = True,
        training: bool = True,
        shuffle_buffer_size: Optional[int] = None,
        after_batch: Optional[Callable] = None,
    ) -> tf.data.Dataset:
        if after_batch is None:
            after_batch = _reshape_x
        return super().batchify(
            batch_size,
            shuffle=shuffle,
            training=training,
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

    def py_text_transform(self, tokens_tensor: tf.Tensor) -> np.ndarray:
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

    def py_transform(self, data: tf.Tensor) -> np.ndarray:
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


def _bert_reshape_x(*data):
    x = data[0]
    reshaped_x = tf.reshape(x, (-1, tf.shape(x)[-1]))
    type_ids = tf.zeros_like(reshaped_x, dtype=tf.int64)
    if len(data) == 1:
        return ((reshaped_x, type_ids),)
    else:
        return ((reshaped_x, type_ids), data[-1])


class BertRetrievalEvaluationDataset(BertDatasetMixin, RetrievalEvaluationDataset):
    @property
    def batch_padding_shapes(self) -> tuple:
        return [(None, None), ()][: None if self.has_label else -1]

    def py_transform_out_dtype(self) -> list[tf.DType]:
        return [self.text_dtype, self.label_dtype][: None if self.has_label else -1]

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
        return [
            tf.keras.preprocessing.sequence.pad_sequences(
                token_ids_list, padding="post"
            )
        ]

    def batchify(
        self,
        batch_size: int,
        shuffle: bool = True,
        training: bool = True,
        shuffle_buffer_size: Optional[int] = None,
        after_batch: Optional[Callable] = None,
    ) -> tf.data.Dataset:
        if after_batch is None:
            after_batch = _bert_reshape_x
        return super().batchify(
            batch_size,
            shuffle=shuffle,
            training=training,
            shuffle_buffer_size=shuffle_buffer_size,
            after_batch=after_batch,
        )