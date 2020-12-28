from typing import Sequence, List, Optional

import numpy as np
import pandas as pd

import tensorflow as tf

from sknlp.vocab import Vocab

from .classification_dataset import ClassificationDataset
from .tagging_dataset import TaggingDataset


class BertClassificationDataset(ClassificationDataset):
    def __init__(
        self,
        vocab: Vocab,
        labels: Sequence[str],
        df: Optional[pd.DataFrame] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        is_multilabel: bool = True,
        max_length: Optional[int] = None,
    ):
        super().__init__(
            vocab,
            labels,
            df=df,
            csv_file=csv_file,
            in_memory=in_memory,
            is_multilabel=is_multilabel,
            max_length=max_length,
            text_segmenter=None,
            text_dtype=tf.string,
            text_padding_shape=(),
            text_padding_value=vocab.pad,
            label_padding_value=0.0,
        )

    def _text_transform(self, text: tf.Tensor) -> str:
        return text.numpy().decode("utf-8")[: self.max_length]

    def batchify(
        self, batch_size: int, shuffle: bool = True, shuffle_buffer_size: int = 100000
    ) -> tf.data.Dataset:
        dataset = self._dataset
        if shuffle:
            if shuffle_buffer_size <= 0:
                shuffle_buffer_size = 100000
            dataset = dataset.shuffle(shuffle_buffer_size)

        return dataset.batch(batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )


class BertTaggingDataset(TaggingDataset):
    def __init__(
        self,
        vocab: Vocab,
        labels: Sequence[str],
        df: Optional[pd.DataFrame] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        max_length: Optional[int] = None,
    ):
        super().__init__(
            vocab,
            labels,
            df=df,
            csv_file=csv_file,
            in_memory=in_memory,
            start_tag="[CLS]",
            end_tag="[SEP]",
            text_segmenter=None,
            max_length=max_length,
            text_dtype=tf.string,
            text_padding_shape=(),
            text_padding_value=vocab.pad,
        )

    def _text_transform(self, text: tf.Tensor) -> str:
        return text.numpy().decode("utf-8")[: self.max_length]