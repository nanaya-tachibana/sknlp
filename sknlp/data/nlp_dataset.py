from __future__ import annotations
from typing import Callable, Optional, Any, Sequence
import string

import pandas as pd
import numpy as np
import tensorflow as tf


class NLPDataset:
    def __init__(
        self,
        tokenizer: Callable[[str], list[int]],
        X: Optional[Sequence[Any]] = None,
        y: Optional[Sequence[Any]] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        no_label: bool = False,
        max_length: Optional[int] = None,
        na_value: str = "",
        column_dtypes: list[str] = ["str", "str"],
        text_dtype: tf.DType = tf.string,
        label_dtype: tf.DType = tf.string,
    ):
        assert (
            X is not None
        ) or csv_file is not None, "either X or csv_file may not be None"
        self.in_memory = in_memory
        self.na_value = na_value
        self.column_dtypes = column_dtypes
        if csv_file is not None:
            self._original_dataset, self.size = self.load_csv(
                csv_file, "\t", in_memory, column_dtypes, na_value
            )
        else:
            no_label = no_label and y is None
            df = self.Xy_to_dataframe(X, y=y)
            self._original_dataset = self.dataframe_to_dataset(
                df, column_dtypes, na_value
            )
            self.size = df.shape[0]

        self.no_label = no_label
        self.tokenizer = tokenizer
        self.max_length = max_length or 99999
        self.text_dtype = text_dtype
        self.label_dtype = label_dtype
        self._dataset = self._transform(self._original_dataset)

    @property
    def X(self) -> list[Any]:
        x = []
        for data in self._original_dataset.as_numpy_iterator():
            texts = data[: None if self.no_label else -1]
            if len(texts) == 1:
                x.append(texts[0].decode("UTF-8"))
            else:
                x.append([text.decode("UTF-8") for text in texts])
        return x

    @property
    def y(self) -> list[str]:
        if self.no_label:
            return []
        return [
            data[-1].decode("UTF-8")
            for data in self._original_dataset.as_numpy_iterator()
        ]

    @property
    def batch_padding_shapes(self) -> Optional[list[tuple]]:
        return None

    def _normalize_X(self, X: Sequence[Any]) -> Sequence[Any]:
        return X

    def _normalize_y(self, y: Sequence[Any]) -> Sequence[Any]:
        return y

    def Xy_to_dataframe(
        self, X: Sequence[Any], y: Optional[Sequence[Any]] = None
    ) -> pd.DataFrame:
        X = self._normalize_X(X)
        if y is not None:
            y = self._normalize_y(y)
        if isinstance(X[0], (list, tuple)):
            df = pd.DataFrame(zip(*X, y) if y is not None else X)
        else:
            df = pd.DataFrame(zip(X, y) if y is not None else X)
        return df

    def _text_transform(self, text: tf.Tensor) -> np.array:
        token_ids = self.tokenizer(text.numpy().decode("UTF-8"))
        return np.array(token_ids[: self.max_length], dtype=np.int32)

    def _label_transform(self, label: tf.Tensor) -> str:
        return label.numpy().decode("UTF-8")

    def _transform_func(self, *data) -> list[Any]:
        text = data[0]
        _text = self._text_transform(text)
        if self.no_label:
            return _text
        label = data[1]
        return _text, self._label_transform(label)

    def _transform_func_out_dtype(self) -> list[tf.DType]:
        return (self.text_dtype, self.label_dtype)[: -1 if self.no_label else None]

    def _transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(
            lambda *data: tf.py_function(
                self._transform_func, inp=data, Tout=self._transform_func_out_dtype()
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    def shuffled_dataset(
        self, shuffle_buffer_size: Optional[int] = None
    ) -> tf.data.Dataset:
        shuffle_buffer_size = shuffle_buffer_size or self.size or 100000
        return self._dataset.shuffle(shuffle_buffer_size)

    def batchify(
        self,
        batch_size: int,
        shuffle: bool = True,
        shuffle_buffer_size: Optional[int] = None,
        before_batch: Optional[Callable] = None,
        after_batch: Optional[Callable] = None,
    ) -> tf.data.Dataset:
        dataset = (
            self.shuffled_dataset(shuffle_buffer_size) if shuffle else self._dataset
        )
        if before_batch is not None:
            dataset = dataset.map(before_batch)
        if self.batch_padding_shapes is None:
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.padded_batch(
                batch_size, padded_shapes=self.batch_padding_shapes
            )
        if after_batch is not None:
            dataset = dataset.map(after_batch)
        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def dataframe_to_dataset(
        self, df: pd.DataFrame, column_dtypes: list[str], na_value: str
    ) -> tf.data.Dataset:
        df.fillna(na_value, inplace=True)
        for dtype, col in zip(column_dtypes, df.columns):
            df[col] = df[col].astype(dtype)
        return tf.data.Dataset.from_tensor_slices(tuple(df[col] for col in df.columns))

    def load_csv(
        self,
        filename: str,
        sep: str,
        in_memory: bool,
        column_dtypes: list[str],
        na_value: str,
    ) -> tuple[tf.data.Dataset, Optional[int]]:
        if in_memory:
            df = pd.read_csv(filename, sep=sep, quoting=3)
            return self.dataframe_to_dataset(df, column_dtypes, na_value), df.shape[0]
        tf_dtype_mapping = {"str": tf.string, "int": tf.int32, "float": tf.float32}
        return (
            tf.data.experimental.CsvDataset(
                filename,
                [
                    tf_dtype_mapping.get(dtype.rstrip(string.digits), "str")
                    for dtype in column_dtypes[: -1 if self.no_label else None]
                ],
                header=True,
                field_delim=sep,
                na_value=na_value,
            ),
            None,
        )