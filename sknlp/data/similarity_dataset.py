from typing import Sequence, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from sknlp.vocab import Vocab
from .nlp_dataset import NLPDataset


def _combine(text, context, label):
    return (text, context), label


class SimilarityDataset(NLPDataset):
    def __init__(
        self,
        vocab: Vocab,
        labels: Sequence[str],
        df: Optional[pd.DataFrame] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        max_length: Optional[int] = None,
        text_segmenter: str = "char",
        text_dtype: tf.DType = tf.int32,
        label_dtype: tf.DType = tf.float32,
        batch_padding_shapes: Optional[Tuple[tf.DType]] = ((None,), (None,), (None,)),
        batch_padding_values: Optional[Tuple[tf.DType]] = (0, 0, 0.0),
    ):
        self.vocab = vocab
        self.label2idx = dict(zip(labels, range(len(labels))))
        super().__init__(
            df=df,
            csv_file=csv_file,
            in_memory=in_memory,
            text_segmenter=text_segmenter,
            max_length=max_length,
            na_value=0.0,
            column_dtypes=["str", "str", "float32"],
            text_dtype=text_dtype,
            label_dtype=label_dtype,
            batch_padding_shapes=batch_padding_shapes,
            batch_padding_values=batch_padding_values,
        )

    @property
    def label(self) -> List[float]:
        return [y for _, y in self._original_dataset.as_numpy_iterator()]

    def _text_transform(self, text: tf.Tensor) -> np.ndarray:
        tokens = super()._text_transform(text)
        return np.array(
            [self.vocab[t] for t in tokens[: self.max_length]], dtype=np.int32
        )

    def _label_transform(self, label: tf.Tensor) -> float:
        return label

    def _transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        def func(text, context, label):
            return (
                self._text_transform(text),
                self._text_transform(context),
                self._label_transform(label),
            )

        return dataset.map(
            lambda t, c, l: tf.py_function(
                func,
                inp=[t, c, l],
                Tout=(self.text_dtype, self.text_dtype, self.label_dtype),
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

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
            after_batch=_combine,
        )