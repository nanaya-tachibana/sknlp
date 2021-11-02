from __future__ import annotations
from typing import Any, Optional, Sequence, Callable

import tensorflow as tf


def _combine_xy(token_ids, type_ids, label):
    return ((token_ids, type_ids), label)


def _combine_x(token_ids, type_ids):
    return ((token_ids, type_ids),)


class BertDatasetMixin:
    @property
    def batch_padding_shapes(self) -> list[tuple]:
        shapes = [(None,), (None,), (None,)]
        return shapes[: None if self.has_label else -1]

    def tf_transform_before_py_transform(
        self, *data: Sequence[tf.Tensor]
    ) -> list[tf.Tensor]:
        tokens = self.tokenize(data[: -1 if self.has_label else None])
        if self.has_label:
            return [tokens, data[-1]]
        return [tokens]

    def py_transform_out_dtype(self) -> list[tf.DType]:
        dtypes = super().py_transform_out_dtype()
        dtypes.insert(0, self.text_dtype)
        return dtypes

    def py_text_transform(self, tokens_tensor: tf.Tensor) -> list[list[int]]:
        cls_id: int = self.vocab["[CLS]"]
        sep_id: int = self.vocab["[SEP]"]
        pad_id: int = self.vocab[self.vocab.pad]
        truncated_token_ids = [cls_id]
        truncated_type_ids = [0]
        for i, tokens in enumerate(tokens_tensor.numpy().tolist()):
            token_ids = self.vocab.token2idx(
                [token.decode("UTF-8") for token in tokens][: self.max_length]
            )
            token_ids = [tid for tid in token_ids if tid != pad_id]
            truncated_token_ids.extend(token_ids)
            truncated_type_ids.extend(i for _ in range(len(token_ids)))
            truncated_token_ids.append(sep_id)
            truncated_type_ids.append(i)
        return [truncated_token_ids, truncated_type_ids]

    def py_transform(self, *data: list[tf.Tensor]) -> list[Any]:
        transformed_data = self.py_text_transform(data[0])
        if self.has_label:
            transformed_data.append(self.py_label_transform(data[-1]))
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
            after_batch = _combine_xy if self.has_label else _combine_x
        return super().batchify(
            batch_size,
            shuffle=shuffle,
            training=training,
            shuffle_buffer_size=shuffle_buffer_size,
            after_batch=after_batch,
        )