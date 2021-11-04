from __future__ import annotations
from typing import Any, Callable, Sequence, Optional

import tensorflow as tf

from sknlp.vocab import Vocab
from .nlp_dataset import NLPDataset
from .bert_mixin import BertDatasetMixin


def _combine_xy(text, target):
    return ((text, target),)


class GenerationDataset(NLPDataset):
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
            segmenter=segmenter,
            X=X,
            y=y,
            csv_file=csv_file,
            in_memory=in_memory,
            has_label=has_label,
            max_length=max_length,
            na_value="",
            column_dtypes=["str", "str"],
            text_dtype=text_dtype,
            label_dtype=label_dtype,
            **kwargs,
        )

    @property
    def batch_padding_shapes(self) -> list[tuple]:
        shapes = [(None,), (None,)]
        return shapes[: None if self.has_label else -1]

    def py_label_transform(self, label: tf.Tensor) -> list[list[int]]:
        return self.py_text_transform(label)

    def normalize_label(self, data: tf.Tensor) -> tf.Tensor:
        return self.normalize_letter_case(data)

    def batchify(
        self,
        batch_size: int,
        shuffle: bool = True,
        training: bool = True,
        after_batch: Optional[
            Callable[[list[tf.Tensor]], list[tf.Tensor] | tf.Tensor]
        ] = None,
        shuffle_buffer_size: Optional[int] = None,
    ) -> tf.data.Dataset:
        if after_batch is None:
            after_batch = _combine_xy if self.has_label and training else None
        return super().batchify(
            batch_size,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            after_batch=after_batch,
        )


class BertGenerationDataset(BertDatasetMixin, GenerationDataset):
    @property
    def batch_padding_shapes(self) -> list[tuple]:
        return [(None,), (None,)]

    def tf_transform_before_py_transform(
        self, *data: Sequence[tf.Tensor]
    ) -> list[tf.Tensor]:
        return [self.tokenize(data)]

    def py_transform(self, *data: list[tf.Tensor]) -> list[Any]:
        return self.py_text_transform(data[0])

    def py_transform_out_dtype(self) -> list[tf.DType]:
        return [self.text_dtype, self.text_dtype]

    def batchify(
        self,
        batch_size: int,
        shuffle: bool = True,
        training: bool = True,
        shuffle_buffer_size: Optional[int] = None,
        after_batch: Optional[Callable] = None,
    ) -> tf.data.Dataset:
        if after_batch is None:
            after_batch = _combine_xy
        return super().batchify(
            batch_size,
            shuffle=shuffle,
            training=training,
            shuffle_buffer_size=shuffle_buffer_size,
            after_batch=after_batch,
        )