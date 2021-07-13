from __future__ import annotations
from typing import Sequence, Optional, Tuple, Callable, Any

import pandas as pd

import tensorflow as tf

from .classification_dataset import ClassificationDataset
from .tagging_dataset import TaggingDataset
from .similarity_dataset import SimilarityDataset


class BertClassificationDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: Callable[[str], list[int]],
        labels: Sequence[str],
        X: Optional[Sequence[Any]] = None,
        y: Optional[Sequence[Any]] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        no_label: bool = False,
        is_multilabel: bool = True,
        max_length: Optional[int] = None,
    ):
        super().__init__(
            tokenizer,
            labels,
            X=X,
            y=y,
            csv_file=csv_file,
            in_memory=in_memory,
            no_label=no_label,
            is_multilabel=is_multilabel,
            max_length=max_length,
            text_dtype=tf.string,
            label_dtype=tf.float32,
        )

    @property
    def batch_padding_shapes(self) -> Optional[list[Tuple]]:
        return None

    def _text_transform(self, text: tf.Tensor) -> str:
        return text.numpy().decode("UTF-8")[: self.max_length]


class BertTaggingDataset(TaggingDataset):
    def __init__(
        self,
        tokenizer: Callable[[str], list[int]],
        labels: Sequence[str],
        X: Optional[Sequence[Any]] = None,
        y: Optional[Sequence[Any]] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        no_label: bool = False,
        add_start_end_tag: bool = True,
        use_crf: bool = False,
        max_length: Optional[int] = None,
    ):
        super().__init__(
            tokenizer,
            labels,
            X=X,
            y=y,
            csv_file=csv_file,
            in_memory=in_memory,
            no_label=no_label,
            use_crf=use_crf,
            add_start_end_tag=add_start_end_tag,
            max_length=max_length,
            text_dtype=tf.string,
            label_dtype=tf.int32,
        )

    @property
    def batch_padding_shapes(self) -> Optional[list[Tuple]]:
        if self.use_crf:
            return ((), (None,))
        else:
            return ((), (None, None, None))

    def _text_transform(self, text: tf.Tensor) -> str:
        return text.numpy().decode("UTF-8")[: self.max_length]


class BertSimilarityDataset(SimilarityDataset):
    def __init__(
        self,
        tokenizer: Callable[[str], list[int]],
        labels: Sequence[str],
        X: Optional[Sequence[Any]] = None,
        y: Optional[Sequence[Any]] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        no_label: bool = False,
        max_length: Optional[int] = None,
    ):
        super().__init__(
            tokenizer,
            labels,
            X=X,
            y=y,
            csv_file=csv_file,
            in_memory=in_memory,
            no_label=no_label,
            max_length=max_length,
            text_dtype=tf.string,
            label_dtype=tf.float32,
        )

    @property
    def batch_padding_shapes(self) -> Optional[list[Tuple]]:
        return None

    def _text_transform(self, text: tf.Tensor) -> str:
        return text.numpy().decode("UTF-8")[: self.max_length]