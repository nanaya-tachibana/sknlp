from typing import Sequence, List, Union, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from sknlp.vocab import Vocab
from .nlp_dataset import NLPDataset


class TaggingDataset(NLPDataset):
    def __init__(
        self,
        vocab: Vocab,
        labels: Sequence[str],
        df: Optional[pd.DataFrame] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        start_tag: Optional[str] = None,
        end_tag: Optional[str] = None,
        max_length: Optional[int] = None,
        text_segmenter: str = "char",
        text_dtype: tf.DType = tf.int32,
        label_dtype: tf.DType = tf.int32,
        text_padding_shape: tuple = (None,),
        label_padding_shape: tuple = (None,),
        text_padding_value: Union[str, int, float, None] = None,
        label_padding_value: Union[str, int, float, None] = 0
    ):
        self.vocab = vocab
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.label2idx = dict(zip(labels, range(len(labels))))
        super().__init__(
            df=df,
            csv_file=csv_file,
            in_memory=in_memory,
            text_segmenter=text_segmenter,
            max_length=max_length,
            text_dtype=text_dtype,
            label_dtype=label_dtype,
            text_padding_shape=text_padding_shape,
            label_padding_shape=label_padding_shape,
            text_padding_value=text_padding_value or vocab[vocab.pad],
            label_padding_value=label_padding_value,
        )

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

    def _combine_xy(self, x, y):
        return (x, y), y

    def batchify(
        self,
        batch_size: int,
        shuffle: bool = True,
        shuffle_buffer_size: int = 100000,
    ) -> tf.data.Dataset:
        dataset = self._dataset
        if shuffle:
            if shuffle_buffer_size <= 0:
                shuffle_buffer_size = 100000
            dataset = dataset.shuffle(shuffle_buffer_size)

        return (
            dataset.padded_batch(
                batch_size,
                padded_shapes=(self.text_padding_shape, self.label_padding_shape),
                padding_values=(self.text_padding_value, self.label_padding_value),
            )
            .map(self._combine_xy)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
