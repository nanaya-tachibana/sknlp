from __future__ import annotations
from typing import Any, Sequence, List, Optional, Tuple, Callable

import pandas as pd
import tensorflow as tf

from .nlp_dataset import NLPDataset


def _combine_xy(text, context, label):
    return ((text, context), label)


def _combine_x(text, context):
    return ((text, context),)


class SimilarityDataset(NLPDataset):
    def __init__(
        self,
        tokenizer: Callable[[str], list[int]],
        labels: Sequence[str],
        X: Optional[Sequence[Any]] = None,
        y: Optional[Sequence[Any]] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        no_label: bool = False,
        max_length: Optional[int] = None,
        text_dtype: tf.DType = tf.int32,
        label_dtype: tf.DType = tf.float32,
    ):
        self.label2idx = dict(zip(labels, range(len(labels))))
        super().__init__(
            tokenizer,
            X=X,
            y=y,
            csv_file=csv_file,
            in_memory=in_memory,
            no_label=no_label,
            max_length=max_length,
            na_value=0.0,
            column_dtypes=["str", "str", "float32"],
            text_dtype=text_dtype,
            label_dtype=label_dtype,
        )

    @property
    def y(self) -> List[float]:
        if self.no_label:
            return []
        return [float(data[-1]) for data in self._original_dataset.as_numpy_iterator()]

    @property
    def batch_padding_shapes(self) -> List[Tuple]:
        return ((None,), (None,), (None,))[: -1 if self.no_label else None]

    def _label_transform(self, label: tf.Tensor) -> float:
        return label

    def _transform_func(self, *data) -> List[Any]:
        text = data[0]
        context = data[1]
        if self.no_label:
            return (
                self._text_transform(text),
                self._text_transform(context),
            )
        label = data[2]
        return (
            self._text_transform(text),
            self._text_transform(context),
            self._label_transform(label),
        )

    def _transform_func_out_dtype(self) -> List[tf.DType]:
        return (self.text_dtype, self.text_dtype, self.label_dtype)[
            : -1 if self.no_label else None
        ]

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
            after_batch=_combine_xy if not self.no_label else _combine_x,
        )