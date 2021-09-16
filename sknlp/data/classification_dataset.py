from __future__ import annotations
from typing import Sequence, Union, Optional, Callable, Any

import numpy as np
import tensorflow as tf

from .nlp_dataset import NLPDataset
from .utils import serialize_example


def _combine_xy(text, context, label):
    return ((text, context), label)


def _combine_x(text, context):
    return ((text, context),)


class ClassificationDataset(NLPDataset):
    def __init__(
        self,
        tokenizer: Callable[[str], list[int]],
        labels: Sequence[str],
        X: Optional[Sequence[Any]] = None,
        y: Optional[Sequence[Any]] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        has_label: bool = True,
        is_multilabel: bool = False,
        is_pair_text: bool = False,
        max_length: Optional[int] = None,
        text_dtype: tf.DType = tf.int32,
        label_dtype: tf.DType = tf.float32,
        **kwargs,
    ):
        self.labels = list(labels)
        self.is_pair_text = is_pair_text
        self.is_multilabel = is_multilabel and not is_pair_text
        column_dtypes = ["str", "str"]
        if self.is_pair_text:
            column_dtypes.append("str")
        self.label2idx = dict(zip(labels, range(len(labels))))
        super().__init__(
            tokenizer,
            X=X,
            y=y,
            csv_file=csv_file,
            in_memory=in_memory,
            has_label=has_label,
            max_length=max_length,
            na_value="NULL",
            column_dtypes=column_dtypes,
            text_dtype=text_dtype,
            label_dtype=label_dtype,
            **kwargs,
        )

    @property
    def y(self) -> Union[list[str], list[list[str]]]:
        if not self.has_label:
            return []
        return [
            data[-1].decode("UTF-8").split("|")
            if self.is_multilabel
            else data[-1].decode("UTF-8")
            for data in self._original_dataset.as_numpy_iterator()
        ]

    @property
    def batch_padding_shapes(self) -> list[tuple]:
        shapes = [(None,), (None,)]
        if self.is_pair_text:
            shapes.append((None,))
        return tuple(shapes[: None if self.has_label else -1])

    def _transform_func(self, *data: list[tf.Tensor]) -> list[Any]:
        num_text_data = self.is_pair_text + 1
        transformed_data = []
        for d in data[:num_text_data]:
            transformed_data.append(self._text_transform(d))
        if self.has_label:
            transformed_data.append(self._label_transform(data[-1]))
        return transformed_data

    def _transform_func_out_dtype(self) -> list[tf.DType]:
        dtypes = [self.text_dtype, self.label_dtype]
        if self.is_pair_text:
            dtypes.insert(0, self.text_dtype)
        return dtypes[: None if self.has_label else -1]

    def _normalize_y(self, y: Sequence[Any]) -> Sequence[Any]:
        if isinstance(y[0], (list, tuple)):
            return ["|".join(map(str, yi)) for yi in y]
        return y

    def _label_binarizer(self, labels: list[str]) -> np.ndarray:
        label2idx = self.label2idx
        res = np.zeros(len(label2idx), dtype=np.float32)
        res[[label2idx[label] for label in labels if label in label2idx]] = 1
        return res

    def _label_transform(self, label: tf.Tensor) -> np.ndarray:
        _label = super()._label_transform(label)
        if self.is_multilabel:
            labels = _label.split("|")
        else:
            labels = [_label]
        return self._label_binarizer(labels)

    def batchify(
        self,
        batch_size: int,
        shuffle: bool = True,
        training: bool = True,
        shuffle_buffer_size: Optional[int] = None,
    ) -> tf.data.Dataset:
        after_batch = None
        if self.is_pair_text:
            if self.has_label:
                after_batch = _combine_xy
            else:
                after_batch = _combine_x
        return super().batchify(
            batch_size,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            after_batch=after_batch,
        )

    def to_tfrecord(self, filename: str) -> None:
        def func(text: np.ndarray, label: np.ndarray):
            return tf.reshape(
                serialize_example(
                    (self._text_transform(text), self._label_transform(label)),
                    ("tensor", "tensor"),
                ),
                (),
            )

        tf_writer = tf.data.experimental.TFRecordWriter(filename)
        tf_writer.write(
            self._dataset.map(
                lambda t, l: tf.py_function(func, inp=[t, l], Tout=tf.string),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        )

    @classmethod
    def from_tfrecord(cls, filename: str) -> tf.data.Dataset:
        def func(record: tf.Tensor):
            parsed_record = tf.io.parse_single_example(
                record,
                {
                    "feature0": tf.io.FixedLenFeature([], tf.string, default_value=""),
                    "feature1": tf.io.FixedLenFeature([], tf.string, default_value=""),
                },
            )
            return (
                tf.io.parse_tensor(parsed_record["feature0"], tf.int32),
                tf.io.parse_tensor(parsed_record["feature1"], tf.float32),
            )

        return tf.data.TFRecordDataset(filename).map(func)