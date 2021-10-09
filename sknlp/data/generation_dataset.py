from __future__ import annotations
from typing import Any, Sequence, Optional

import tensorflow as tf
import numpy as np

from sknlp.vocab import Vocab
from .nlp_dataset import NLPDataset


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
        text_dtype: tf.DType = tf.int32,
        label_dtype: tf.DType = tf.int32,
        **kwargs
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
            **kwargs
        )

    @property
    def batch_padding_shapes(self) -> list[tuple]:
        return (
            (None,),
            (None,),
        )[: None if self.has_label else -1]

    def _label_transform(self, label: tf.Tensor) -> np.array:
        return self._text_transform(label)

    def batchify(
        self,
        batch_size: int,
        shuffle: bool = True,
        training: bool = True,
        shuffle_buffer_size: Optional[int] = None,
    ) -> tf.data.Dataset:
        return super().batchify(
            batch_size,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            after_batch=_combine_xy if self.has_label and training else None,
        )