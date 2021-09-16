from __future__ import annotations
from typing import Sequence, Optional, Callable, Any

import tensorflow as tf

from .classification_dataset import ClassificationDataset
from .tagging_dataset import TaggingDataset
from .generation_dataset import GenerationDataset


class BertGenerationDataset(GenerationDataset):
    def __init__(
        self,
        tokenizer: Callable[[str], list[int]],
        labels: Sequence[str],
        X: Optional[Sequence[Any]] = None,
        y: Optional[Sequence[Any]] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        has_label: bool = True,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer,
            labels,
            X=X,
            y=y,
            csv_file=csv_file,
            in_memory=in_memory,
            has_label=has_label,
            max_length=max_length,
            text_dtype=tf.string,
            label_dtype=tf.string,
            **kwargs,
        )

    @property
    def batch_padding_shapes(self) -> Optional[list]:
        return None

    def _text_transform(self, text: tf.Tensor) -> str:
        return text.numpy().decode("UTF-8")[: self.max_length]


class BertClassificationDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: Callable[[str], list[int]],
        labels: Sequence[str],
        X: Optional[Sequence[Any]] = None,
        y: Optional[Sequence[Any]] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        has_label: bool = True,
        is_multilabel: bool = True,
        is_pair_text: bool = False,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer,
            labels,
            X=X,
            y=y,
            csv_file=csv_file,
            in_memory=in_memory,
            has_label=has_label,
            is_multilabel=is_multilabel,
            is_pair_text=is_pair_text,
            max_length=max_length,
            text_dtype=tf.string,
            label_dtype=tf.float32,
            **kwargs,
        )

    @property
    def batch_padding_shapes(self) -> None:
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
        has_label: bool = True,
        add_start_end_tag: bool = True,
        output_format: str = "global_pointer",
        max_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer,
            labels,
            X=X,
            y=y,
            csv_file=csv_file,
            in_memory=in_memory,
            has_label=has_label,
            output_format=output_format,
            add_start_end_tag=add_start_end_tag,
            max_length=max_length,
            text_dtype=tf.string,
            label_dtype=tf.int32,
            **kwargs,
        )

    @property
    def batch_padding_shapes(self) -> Optional[list[tuple]]:
        if self.output_format == "bio":
            return ((), (None,))
        else:
            return ((), (None, None, None))

    def _text_transform(self, text: tf.Tensor) -> str:
        return text.numpy().decode("UTF-8")[: self.max_length]