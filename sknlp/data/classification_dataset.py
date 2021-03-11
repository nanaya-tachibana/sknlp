from typing import Sequence, List, Union, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from sknlp.vocab import Vocab
from .nlp_dataset import NLPDataset
from .utils import serialize_example


class ClassificationDataset(NLPDataset):
    def __init__(
        self,
        vocab: Vocab,
        labels: Sequence[str],
        df: Optional[pd.DataFrame] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        no_label: bool = False,
        is_multilabel: bool = True,
        max_length: Optional[int] = None,
        text_segmenter: str = "char",
        text_dtype: tf.DType = tf.int32,
        label_dtype: tf.DType = tf.float32,
    ):
        self.vocab = vocab
        self.labels = list(labels)
        self.is_multilabel = is_multilabel
        self.label2idx = dict(zip(labels, range(len(labels))))
        super().__init__(
            df=df,
            csv_file=csv_file,
            in_memory=in_memory,
            no_label=no_label,
            text_segmenter=text_segmenter,
            max_length=max_length,
            na_value="" if is_multilabel else "NULL",
            column_dtypes=["str", "str"],
            text_dtype=text_dtype,
            label_dtype=label_dtype,
        )

    @property
    def y(self) -> Union[List[str], List[List[str]]]:
        if self.no_label:
            return []
        return [
            data[-1].decode("utf-8").split("|")
            if self.is_multilabel
            else data[-1].decode("Utf-8")
            for data in self._original_dataset.as_numpy_iterator()
        ]

    @property
    def batch_padding_shapes(self) -> List[Tuple]:
        return ((None,), (None,))[: -1 if self.no_label else None]

    def _text_transform(self, text: tf.Tensor) -> np.ndarray:
        tokens = super()._text_transform(text)
        return np.array(
            [self.vocab[t] for t in tokens[: self.max_length]], dtype=np.int32
        )

    def _label_binarizer(self, labels: List[str]) -> np.ndarray:
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
