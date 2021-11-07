from __future__ import annotations
from typing import Any, Sequence, Optional

import tensorflow as tf

from sknlp.vocab import Vocab
from sknlp.utils.tensor import pad2shape
from .nlp_dataset import NLPDataset


def _combine_xy(text, context, label):
    return tf.cond(
        tf.shape(text)[1] > tf.shape(context)[1],
        lambda: (
            tf.concat([text, pad2shape(context, tf.shape(text))], 0),
            tf.squeeze(label),
        ),
        lambda: (
            tf.concat([pad2shape(text, tf.shape(context)), context], 0),
            tf.squeeze(label),
        ),
    )


def _combine_x(text, context):
    return tf.cond(
        tf.shape(text)[1] > tf.shape(context)[1],
        lambda: tf.concat([text, pad2shape(context, tf.shape(text))], 0),
        lambda: tf.concat([pad2shape(text, tf.shape(context)), context], 0),
    )


class SimilarityDataset(NLPDataset):
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
        label_dtype: tf.DType = tf.float32,
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
            na_value=0.0,
            column_dtypes=["str", "str", "float32"],
            text_dtype=text_dtype,
            label_dtype=label_dtype,
        )

    @property
    def y(self) -> list[float]:
        if not self.has_label:
            return []
        return [float(data[-1]) for data in self._original_dataset.as_numpy_iterator()]

    @property
    def batch_padding_shapes(self) -> list[tuple]:
        return ((None,), (None,), ())[: None if self.has_label else -1]

    def _label_transform(self, label: tf.Tensor) -> float:
        return label

    def _transform_func(self, *data) -> list[Any]:
        text = data[0]
        context = data[1]
        if not self.has_label:
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

    def _transform_func_out_dtype(self) -> list[tf.DType]:
        dtypes = super()._transform_func_out_dtype()
        dtypes.insert(0, self.text_dtype)
        return dtypes

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
            after_batch=_combine_xy if self.has_label else _combine_x,
        )


def _combine_string_xy(text, context, label):
    return tf.concat([text, context], 0), tf.squeeze(label)


def _combine_string_x(text, context):
    return tf.concat([text, context], 0)


class BertSimilarityDataset(SimilarityDataset):
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
            text_dtype=tf.string,
            label_dtype=tf.float32,
            **kwargs,
        )

    @property
    def batch_padding_shapes(self) -> Optional[list]:
        return None

    def _text_transform(self, text: tf.Tensor) -> str:
        return text.numpy().decode("UTF-8").lower()[: self.max_length]

    def batchify(
        self,
        batch_size: int,
        shuffle: bool = True,
        training: bool = True,
        shuffle_buffer_size: Optional[int] = None,
    ) -> tf.data.Dataset:
        return super(SimilarityDataset, self).batchify(
            batch_size,
            shuffle=shuffle,
            training=training,
            shuffle_buffer_size=shuffle_buffer_size,
            after_batch=_combine_string_xy if self.has_label else _combine_string_x,
        )