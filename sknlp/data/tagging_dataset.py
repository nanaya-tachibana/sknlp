from __future__ import annotations
from typing import Sequence, Optional, Tuple, Any, Callable
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from .nlp_dataset import NLPDataset


def _combine_xy(x, y):
    return (x, y), y


def _flatten_y(x, y):
    y_shape = tf.shape(y)
    return x, tf.reshape(y, [y_shape[0], y_shape[1], -1])


class TaggingDataset(NLPDataset):
    def __init__(
        self,
        tokenizer: Callable[[str], list[int]],
        labels: Sequence[str],
        X: Optional[Sequence[Any]] = None,
        y: Optional[Sequence[Any]] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        no_label: bool = False,
        add_start_end_tag: bool = False,
        use_crf: bool = False,
        max_length: Optional[int] = None,
        text_dtype: tf.DType = tf.int32,
        label_dtype: tf.DType = tf.int32,
    ):
        self.add_start_end_tag = add_start_end_tag
        self.use_crf = use_crf
        self.label2idx = dict(zip(labels, range(len(labels))))
        super().__init__(
            tokenizer,
            X=X,
            y=y,
            csv_file=csv_file,
            in_memory=in_memory,
            no_label=no_label,
            max_length=max_length,
            na_value="",
            column_dtypes=["str", "str"],
            text_dtype=text_dtype,
            label_dtype=label_dtype,
        )

    @property
    def y(self) -> list[list[str]]:
        if self.no_label:
            return []
        return [
            json.loads(data[-1].decode("UTF-8"))
            for data in self._original_dataset.as_numpy_iterator()
        ]

    @property
    def batch_padding_shapes(self) -> list[Tuple]:
        if self.use_crf:
            return ((None,), (None,))
        else:
            return ((None,), (None, None, None))

    def _normalize_y(self, y: Sequence[Any]) -> Sequence[Any]:
        if isinstance(y[0], (list, tuple)):
            return [json.dumps(yi) for yi in y]
        return y

    def _label_transform(self, label: tf.Tensor, length: int) -> np.array:
        label = super()._label_transform(label)
        chunks = json.loads(label)
        length += 2 * self.add_start_end_tag
        if self.use_crf:
            labels = np.zeros(length, dtype=np.int32)
            for chunk_start, chunk_end, chunk_label in chunks:
                if chunk_end >= self.max_length:
                    continue

                chunk_start += self.add_start_end_tag
                chunk_end += self.add_start_end_tag
                labels[chunk_start] = self.label2idx[chunk_label] * 2 - 1
                for i in range(chunk_start + 1, chunk_end + 1):
                    labels[i] = self.label2idx[chunk_label] * 2
        else:
            labels = np.zeros((len(self.label2idx), length, length), dtype=np.int32)
            for chunk_start, chunk_end, chunk_label in chunks:
                if chunk_end > self.max_length:
                    continue
                labels[self.label2idx[chunk_label], chunk_start, chunk_end] = 1
        return labels

    def _transform_func(self, *data) -> list[Any]:
        text = data[0]

        _text = self._text_transform(text)
        if self.no_label:
            if self.use_crf:
                return _text, [0 for _ in range(len(_text))]
            else:
                return _text
        label = data[1]
        return _text, self._label_transform(label, len(_text))

    def _transform_func_out_dtype(self) -> list[tf.DType]:
        return (self.text_dtype, self.label_dtype)

    def batchify(
        self,
        batch_size: int,
        shuffle: bool = True,
        shuffle_buffer_size: Optional[int] = None,
    ) -> tf.data.Dataset:
        return super().batchify(
            batch_size,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            after_batch=_combine_xy if self.use_crf else _flatten_y,
        )
