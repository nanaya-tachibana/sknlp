from __future__ import annotations
from typing import Sequence, List, Optional, Tuple, Any
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sknlp.vocab import Vocab
from .nlp_dataset import NLPDataset


def _combine_xy(x, y):
    return (x, y), y


def _flatten_y(x, y):
    y_shape = tf.shape(y)
    return x, tf.reshape(y, [y_shape[0], y_shape[1], -1])


class TaggingDataset(NLPDataset):
    def __init__(
        self,
        vocab: Vocab,
        labels: Sequence[str],
        df: Optional[pd.DataFrame] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        no_label: bool = False,
        use_crf: bool = False,
        start_tag: Optional[str] = None,
        end_tag: Optional[str] = None,
        max_length: Optional[int] = None,
        text_segmenter: str = "char",
        text_dtype: tf.DType = tf.int32,
        label_dtype: tf.DType = tf.int32,
    ):
        self.vocab = vocab
        self.use_crf = use_crf
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
            json.loads(data[-1].decode("utf-8"))
            for data in self._original_dataset.as_numpy_iterator()
        ]

    @property
    def batch_padding_shapes(self) -> List[Tuple]:
        if self.use_crf:
            return ((None,), (None,))
        else:
            return ((None,), (None, None, None))

    def _text_transform(self, text: tf.Tensor) -> np.ndarray:
        tokens = super()._text_transform(text)
        return np.array([self.vocab[t] for t in tokens], dtype=np.int32)

    def _label_transform(self, label: tf.Tensor, length: int) -> np.array:
        label = super()._label_transform(label)
        chunks = json.loads(label)
        add_start_end_tag = self.start_tag is not None and self.end_tag is not None
        max_end_idx = length + 2 * add_start_end_tag
        if self.use_crf:
            labels = np.full(max_end_idx, self.label2idx["O"], dtype=np.int32)
            if add_start_end_tag:
                labels[0] = self.label2idx[self.start_tag]
                labels[-1] = self.label2idx[self.end_tag]
            for chunk_start, chunk_end, chunk_label in chunks:
                if chunk_end >= length:
                    continue

                chunk_start += add_start_end_tag
                chunk_end += add_start_end_tag
                labels[chunk_start] = self.label2idx["-".join(["B", chunk_label])]
                for i in range(chunk_start + 1, chunk_end + 1):
                    labels[i] = self.label2idx["-".join(["I", chunk_label])]
        else:
            labels = np.zeros(
                (len(self.label2idx), max_end_idx, max_end_idx), dtype=np.int32
            )
            for chunk_start, chunk_end, chunk_label in chunks:
                if chunk_end > self.max_length:
                    continue
                chunk_start += add_start_end_tag
                chunk_end += add_start_end_tag
                labels[self.label2idx[chunk_label], chunk_start, chunk_end] = 1
        return labels

    def _transform_func(self, *data) -> List[Any]:
        text = data[0]

        _text = self._text_transform(text)
        if self.no_label:
            if self.use_crf:
                return _text, [0 for _ in range(len(_text))]
            else:
                return _text
        label = data[1]
        return _text, self._label_transform(label, len(_text))

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
            after_batch=_combine_xy if self.use_crf else _flatten_y,
        )
