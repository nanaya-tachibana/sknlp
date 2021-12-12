from __future__ import annotations
from typing import Callable, Sequence, Optional, Tuple, Any
import json

import numpy as np
import tensorflow as tf

from sknlp.vocab import Vocab
from .nlp_dataset import NLPDataset
from .bert_mixin import BertDatasetMixin


def _combine_xy(x, y):
    return ((x, y),)


class TaggingDataset(NLPDataset):
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
        add_start_end_tag: bool = False,
        output_format: str = "global_pointer",
        max_length: Optional[int] = None,
        text_dtype: tf.DType = tf.int64,
        label_dtype: tf.DType = tf.int32,
        **kwargs,
    ):
        self.add_start_end_tag = add_start_end_tag
        self.output_format = output_format
        self.label2idx = dict(zip(labels, range(len(labels))))
        super().__init__(
            vocab,
            segmenter=segmenter,
            X=X,
            y=y,
            csv_file=csv_file,
            in_memory=in_memory,
            has_label=has_label,
            max_length=max_length,
            na_value="",
            column_dtypes=["str", "str"],
            text_dtype=text_dtype,
            label_dtype=label_dtype,
            **kwargs,
        )

    @property
    def y(self) -> list[list[str]]:
        if not self.has_label:
            return []
        return [
            json.loads(data[-1].decode("UTF-8"))
            for data in self._original_dataset.as_numpy_iterator()
        ]

    @property
    def batch_padding_shapes(self) -> list[Tuple]:
        if self.output_format == "bio":
            return [(None,), (None,)][: None if self.has_label else -1]
        else:
            return [(None,), (None, None, None)][: None if self.has_label else -1]

    def _format_y(self, y: Sequence[Any]) -> list[Sequence[Any]]:
        if isinstance(y[0], (list, tuple)):
            y = [json.dumps(yi) for yi in y]
        return [y]

    def py_label_transform(self, label: tf.Tensor, tokens: Sequence[str]) -> np.ndarray:
        label = super().py_label_transform(label)
        length = min(len(tokens), self.max_length)
        length += 2 * self.add_start_end_tag

        start_mapping, end_mapping = self.vocab.create_ichar2itoken_mapping(tokens)
        chunks = json.loads(label)
        if self.output_format == "bio":
            labels = np.zeros(length, dtype=np.int32)
            for chunk_start, chunk_end, chunk_label in chunks:
                chunk_start = start_mapping[chunk_start]
                chunk_end = end_mapping[chunk_end]
                if (
                    chunk_start == -1
                    or chunk_end == -1
                    or chunk_end >= length - 2 * self.add_start_end_tag
                ):
                    continue

                chunk_start += self.add_start_end_tag
                chunk_end += self.add_start_end_tag
                labels[chunk_start] = self.label2idx[chunk_label] * 2 - 1
                for i in range(chunk_start + 1, chunk_end + 1):
                    labels[i] = self.label2idx[chunk_label] * 2
        else:
            labels = np.zeros((len(self.label2idx), length, length), dtype=np.int32)
            for chunk_start, chunk_end, chunk_label in chunks:
                chunk_start = start_mapping[chunk_start]
                chunk_end = end_mapping[chunk_end]
                if (
                    chunk_start == -1
                    or chunk_end == -1
                    or chunk_end >= length - 2 * self.add_start_end_tag
                ):
                    continue

                chunk_start += self.add_start_end_tag
                chunk_end += self.add_start_end_tag
                labels[self.label2idx[chunk_label], chunk_start, chunk_end] = 1
        return labels

    def py_transform(self, *data: Sequence[tf.Tensor]) -> list[Any]:
        tokens_list = self.py_text_transform(data[: -1 if self.has_label else None])
        transformed_data = self.vocab.token2idx(tokens_list)
        if self.has_label:
            # TODO: 仅在最后一句上做序列标注
            transformed_data.append(self.py_label_transform(data[-1], tokens_list[-1]))
        return transformed_data

    def batchify(
        self,
        batch_size: int,
        shuffle: bool = True,
        training: bool = True,
        shuffle_buffer_size: Optional[int] = None,
        after_batch: Optional[Callable] = None,
    ) -> tf.data.Dataset:
        if (
            after_batch is None
            and self.output_format == "bio"
            and self.has_label
            and training
        ):
            after_batch = _combine_xy
        return super().batchify(
            batch_size,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            after_batch=after_batch,
        )


def _combine_xyz(x, y, z):
    return ((x, y, z),)


def _combine_xy_z(x, y, z):
    return ((x, y), z)


class BertTaggingDataset(BertDatasetMixin, TaggingDataset):
    @property
    def batch_padding_shapes(self) -> list[tuple]:
        shapes = super().batch_padding_shapes
        if self.output_format != "bio" and self.has_label:
            shapes[-1] = (None, None, None)
        return shapes

    def py_transform(self, *data: list[tf.Tensor]) -> list[Any]:
        tokens = [token.decode("UTF-8") for token in data[0].numpy().tolist()[-1]]
        transformed_data = self.py_text_transform(data[0])
        if self.has_label:
            # TODO: 仅在最后一句上做序列标注
            transformed_data.append(self.py_label_transform(data[-1], tokens))
        return transformed_data

    def batchify(
        self,
        batch_size: int,
        shuffle: bool = True,
        training: bool = True,
        shuffle_buffer_size: Optional[int] = None,
        after_batch: Optional[Callable] = None,
    ) -> tf.data.Dataset:
        if after_batch is None:
            if not self.has_label:
                after_batch = _combine_xy
            elif self.output_format == "bio":
                if training:
                    after_batch = _combine_xyz
                else:
                    after_batch = _combine_xy_z
            else:
                after_batch = _combine_xy_z
        return super().batchify(
            batch_size,
            shuffle=shuffle,
            training=training,
            shuffle_buffer_size=shuffle_buffer_size,
            after_batch=after_batch,
        )