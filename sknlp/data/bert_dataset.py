from typing import Sequence, List, Optional, Tuple

import numpy as np
import pandas as pd

import tensorflow as tf

from sknlp.vocab import Vocab

from .classification_dataset import ClassificationDataset
from .tagging_dataset import TaggingDataset
from .similarity_dataset import SimilarityDataset


class BertClassificationDataset(ClassificationDataset):
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
    ):
        super().__init__(
            vocab,
            labels,
            df=df,
            csv_file=csv_file,
            in_memory=in_memory,
            no_label=no_label,
            is_multilabel=is_multilabel,
            max_length=max_length,
            text_segmenter=None,
            text_dtype=tf.string,
            label_dtype=tf.float32,
        )

    @property
    def batch_padding_shapes(self) -> Optional[List[Tuple]]:
        return None

    def _text_transform(self, text: tf.Tensor) -> str:
        return text.numpy().decode("utf-8")[: self.max_length]


class BertTaggingDataset(TaggingDataset):
    def __init__(
        self,
        vocab: Vocab,
        labels: Sequence[str],
        df: Optional[pd.DataFrame] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        no_label: bool = False,
        max_length: Optional[int] = None,
    ):
        super().__init__(
            vocab,
            labels,
            df=df,
            csv_file=csv_file,
            in_memory=in_memory,
            no_label=no_label,
            start_tag="[CLS]",
            end_tag="[SEP]",
            text_segmenter=None,
            max_length=max_length,
            text_dtype=tf.string,
            label_dtype=tf.int32,
        )

    @property
    def batch_padding_shapes(self) -> Optional[List[Tuple]]:
        return ((), (None,))

    def _text_transform(self, text: tf.Tensor) -> str:
        return text.numpy().decode("utf-8")[: self.max_length]


class BertSimilarityDataset(SimilarityDataset):
    def __init__(
        self,
        vocab: Vocab,
        labels: Sequence[str],
        df: Optional[pd.DataFrame] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        no_label: bool = False,
        max_length: Optional[int] = None,
    ):
        super().__init__(
            vocab,
            labels,
            df=df,
            csv_file=csv_file,
            in_memory=in_memory,
            no_label=no_label,
            max_length=max_length,
            text_segmenter=None,
            text_dtype=tf.string,
            label_dtype=tf.float32,
        )

    @property
    def batch_padding_shapes(self) -> Optional[List[Tuple]]:
        return None

    def _text_transform(self, text: tf.Tensor) -> str:
        return text.numpy().decode("utf-8")[: self.max_length]