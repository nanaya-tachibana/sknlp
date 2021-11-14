from __future__ import annotations
from typing import Callable, Optional, Any, Sequence
import string

import pandas as pd
import tensorflow as tf

from sknlp.vocab import Vocab
from .tokenizer import get_tokenizer


class NLPDataset:
    def __init__(
        self,
        vocab: Vocab,
        segmenter: Optional[str] = None,
        X: Optional[Sequence[str] | Sequence[Sequence[str]]] = None,
        y: Optional[Sequence[str | float | int]] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        has_label: bool = True,
        text_normalization: dict[str, str] = {"letter_case": "lowercase"},
        max_length: Optional[int] = None,
        na_value: str = "",
        column_dtypes: list[str] = ["str", "str"],
        text_dtype: tf.DType = tf.string,
        label_dtype: tf.DType = tf.string,
        **kwargs
    ) -> None:
        if X is None and csv_file is None:
            raise ValueError("Either `X` or `csv_file` may not be None.")
        self.in_memory = in_memory
        self.na_value = na_value
        self.text_normalization = text_normalization
        self.column_dtypes = column_dtypes
        self.tokenize = get_tokenizer(segmenter, vocab, max_length=max_length).tokenize
        self.vocab = vocab
        self.max_length = max_length or 99999
        self.text_dtype = text_dtype
        self.label_dtype = label_dtype

        self._has_label = has_label
        if csv_file is not None:
            self._original_dataset, self.size = self.load_csv(
                csv_file,
                "\t",
                in_memory,
                column_dtypes,
                na_value,
                has_label=self._has_label,
            )
        else:
            self._has_label = has_label and y is not None
            df = self.Xy_to_dataframe(X, y=y)
            self._original_dataset = self.dataframe_to_dataset(
                df, column_dtypes, na_value
            )
            self.size = df.shape[0]
        self._original_dataset = self._original_dataset.map(
            self.normalize, num_parallel_calls=tf.data.AUTOTUNE
        )

    @property
    def has_label(self) -> bool:
        return self._has_label

    @property
    def X(self) -> list[Any]:
        x = []
        for data in self._original_dataset.as_numpy_iterator():
            texts = data[: -1 if self.has_label else None]
            if len(texts) == 1:
                x.append(texts[0].decode("UTF-8"))
            else:
                x.append([text.decode("UTF-8") for text in texts])
        return x

    @property
    def y(self) -> list[str]:
        if not self.has_label:
            return []
        return [
            data[-1].decode("UTF-8")
            for data in self._original_dataset.as_numpy_iterator()
        ]

    @property
    def batch_padding_shapes(self) -> list[tuple] | tuple | None:
        return None

    def normalize_letter_case(self, data: tf.Tensor) -> tf.Tensor:
        letter_case = self.text_normalization["letter_case"]
        if letter_case == "lowercase":
            return tf.strings.lower(data, encoding="utf-8")
        elif letter_case == "uppercase":
            return tf.strings.upper(data, encoding="utf-8")
        return data

    def normalize_text(self, *data: list[tf.Tensor]) -> list[tf.Tensor]:
        return [self.normalize_letter_case(d) for d in data]

    def normalize_label(self, data: tf.Tensor) -> tf.Tensor:
        return data

    def normalize(self, *data: list[tf.Tensor]) -> list[tf.Tensor]:
        normalized = self.normalize_text(*data[: -1 if self.has_label else None])
        if self.has_label:
            normalized.append(self.normalize_label(data[-1]))
        return normalized

    def _format_X(self, X: Sequence[Sequence[str]]) -> list[Sequence[str]]:
        if isinstance(X[0], str):
            return [X]

        return list(zip(*X))

    def _format_y(self, y: Sequence[Any]) -> list[Sequence[Any]]:
        return [y]

    def Xy_to_dataframe(
        self,
        X: Sequence[str] | Sequence[Sequence[str]],
        y: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        X = self._format_X(X)
        if y is not None:
            y = self._format_y(y)
        return pd.DataFrame(zip(*X, *y) if y is not None else zip(*X))

    def py_text_transform(
        self, text: tf.Tensor | Sequence[tf.Tensor]
    ) -> list[list[str]]:
        if isinstance(text, tf.Tensor):
            text = [text]
        return [self.tokenize(t.numpy().decode("UTF-8")) for t in text]

    def py_label_transform(self, label: tf.Tensor) -> str:
        return label.numpy().decode("UTF-8")

    def py_transform(self, *data: Sequence[tf.Tensor]) -> list[Any]:
        transformed_data = self.vocab.token2idx(
            self.py_text_transform(data[: -1 if self.has_label else None])
        )
        if self.has_label:
            transformed_data.append(self.py_label_transform(data[-1]))
        return transformed_data

    def py_transform_out_dtype(self) -> list[tf.DType] | tf.DType:
        return [self.text_dtype, self.label_dtype][: None if self.has_label else -1]

    def tf_transform_before_py_transform(
        self, *data: Sequence[tf.Tensor]
    ) -> list[tf.Tensor]:
        return data

    def tf_transform_after_py_transform(
        self, *data: Sequence[tf.Tensor]
    ) -> list[tf.Tensor]:
        return data

    def shuffled_dataset(
        self, dataset: tf.data.Dataset, shuffle_buffer_size: Optional[int] = None
    ) -> tf.data.Dataset:
        shuffle_buffer_size = shuffle_buffer_size or self.size or 100000
        return dataset.shuffle(shuffle_buffer_size)

    def batchify(
        self,
        batch_size: int,
        shuffle: bool = True,
        training: bool = True,
        shuffle_buffer_size: Optional[int] = None,
        after_batch: Optional[Callable] = None,
    ) -> tf.data.Dataset:
        dataset = self._original_dataset
        dataset = dataset.map(
            self.tf_transform_before_py_transform, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.map(
            lambda *data: tf.py_function(
                self.py_transform,
                inp=data,
                Tout=self.py_transform_out_dtype(),
            )
        ).cache()
        dataset = dataset.map(
            self.tf_transform_after_py_transform, num_parallel_calls=tf.data.AUTOTUNE
        )
        if shuffle:
            dataset = self.shuffled_dataset(
                dataset, shuffle_buffer_size=shuffle_buffer_size
            )
        if self.batch_padding_shapes is None:
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.padded_batch(
                batch_size, padded_shapes=tuple(self.batch_padding_shapes)
            )
        if after_batch is not None:
            dataset = dataset.map(
                after_batch,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    @classmethod
    def dataframe_to_dataset(
        cls, df: pd.DataFrame, column_dtypes: list[str], na_value: str
    ) -> tf.data.Dataset:
        df.fillna(na_value, inplace=True)
        for dtype, col in zip(column_dtypes, df.columns):
            df[col] = df[col].astype(dtype)
        series = [df[col] for col in df.columns]
        return tf.data.Dataset.from_tensor_slices(tuple(series))

    @classmethod
    def load_csv(
        cls,
        filename: str,
        sep: str,
        in_memory: bool,
        column_dtypes: list[str],
        na_value: str,
        has_label: bool = True,
    ) -> tuple[tf.data.Dataset, Optional[int]]:
        if in_memory:
            df = pd.read_csv(filename, sep=sep, quoting=3, escapechar="\\")
            return cls.dataframe_to_dataset(df, column_dtypes, na_value), df.shape[0]
        tf_dtype_mapping = {"str": tf.string, "int": tf.int32, "float": tf.float32}
        return (
            tf.data.experimental.CsvDataset(
                filename,
                [
                    tf_dtype_mapping.get(dtype.rstrip(string.digits), "str")
                    for dtype in column_dtypes[: None if has_label else -1]
                ],
                header=True,
                field_delim=sep,
                na_value=na_value,
            ),
            None,
        )