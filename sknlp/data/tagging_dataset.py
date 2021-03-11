from typing import Sequence, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import tensorflow as tf

from sknlp.vocab import Vocab
from .nlp_dataset import NLPDataset


def _combine_xy(x, y):
    return (x, y), y


class TaggingDataset(NLPDataset):
    def __init__(
        self,
        vocab: Vocab,
        labels: Sequence[str],
        df: Optional[pd.DataFrame] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        no_label: bool = False,
        start_tag: Optional[str] = None,
        end_tag: Optional[str] = None,
        max_length: Optional[int] = None,
        text_segmenter: str = "char",
        text_dtype: tf.DType = tf.int32,
        label_dtype: tf.DType = tf.int32,
    ):
        self.vocab = vocab
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.label2idx = dict(zip(labels, range(len(labels))))
        super().__init__(
            df=df,
            csv_file=csv_file,
            in_memory=in_memory,
            no_label=no_label,
            text_segmenter=text_segmenter,
            max_length=max_length,
            na_value="",
            column_dtypes=["str", "str"],
            text_dtype=text_dtype,
            label_dtype=label_dtype,
        )

    @property
    def y(self) -> List[List[str]]:
        if self.no_label:
            return []
        return [
            data[-1].decode("utf-8").split("|")
            for data in self._original_dataset.as_numpy_iterator()
        ]

    @property
    def batch_padding_shapes(self) -> List[Tuple]:
        return ((None,), (None,))

    def _text_transform(self, text: tf.Tensor) -> np.ndarray:
        tokens = super()._text_transform(text)
        return np.array(
            [self.vocab[t] for t in tokens[: self.max_length]], dtype=np.int32
        )

    def _label_transform(self, label: tf.Tensor) -> List[int]:
        label = super()._label_transform(label)
        labels = [self.label2idx[l] for l in label.split("|")][: self.max_length]
        if self.start_tag is not None and self.end_tag is not None:
            labels = [
                self.label2idx[self.start_tag],
                *labels,
                self.label2idx[self.end_tag],
            ]
        return labels

    def _transform_func(self, *data) -> List[Any]:
        text = data[0]
        if self.no_label:
            _text = self._text_transform(text)
            return _text, [0 for _ in range(len(_text))]
        label = data[1]
        return self._text_transform(text), self._label_transform(label)

    def _transform_func_out_dtype(self) -> List[tf.DType]:
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
            after_batch=_combine_xy,
        )
