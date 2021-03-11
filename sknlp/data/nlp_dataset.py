from typing import Callable, Optional, List, Union, Sequence, Tuple, Any
import string

import pandas as pd
import tensorflow as tf
from .text_segmenter import get_segmenter


class NLPDataset:
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        no_label: bool = False,
        text_segmenter: str = "char",
        max_length: Optional[int] = None,
        na_value: str = "",
        column_dtypes: List[str] = ["str", "str"],
        text_dtype: tf.DType = tf.string,
        label_dtype: tf.DType = tf.string,
    ):
        assert (
            df is not None or csv_file is not None
        ), "either df or csv_file may not be None"
        self.in_memory = in_memory
        self.na_value = na_value
        self.column_dtypes = column_dtypes
        if df is not None:
            self._original_dataset = self.dataframe_to_dataset(df, na_value)
            self.size = df.shape[0]
        elif csv_file is not None:
            self._original_dataset, self.size = self.load_csv(
                csv_file, "\t", in_memory, column_dtypes, na_value
            )
        self.no_label = no_label
        self.text_cutter = get_segmenter(text_segmenter)
        self.max_length = max_length or 99999
        self.text_dtype = text_dtype
        self.label_dtype = label_dtype
        self._dataset = self._transform(self._original_dataset)

    @property
    def X(self) -> List[Any]:
        x = []
        for data in self._original_dataset.as_numpy_iterator():
            texts = data[:-1]
            if len(texts) == 1:
                x.append(texts[0].decode("utf-8"))
            else:
                x.append([text.decode("utf-8") for text in texts])
        return x

    @property
    def batch_padding_shapes(self) -> Optional[List[Tuple]]:
        return None

    @property
    def y(self) -> List[str]:
        if self.no_label:
            return []
        return [
            data[-1].decode("utf-8")
            for data in self._original_dataset.as_numpy_iterator()
        ]

    def _text_transform(self, text: tf.Tensor) -> Union[List[str], str]:
        return self.text_cutter(text.numpy().decode("utf-8"))[: self.max_length]

    def _label_transform(self, label: tf.Tensor) -> str:
        return label.numpy().decode("utf-8")

    def _transform_func(self, *data) -> List[Any]:
        text = data[0]
        if self.no_label:
            return self._text_transform(text)
        label = data[1]
        return self._text_transform(text), self._label_transform(label)

    def _transform_func_out_dtype(self) -> List[tf.DType]:
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

    def dataframe_to_dataset(self, df: pd.DataFrame, na_value: str) -> tf.data.Dataset:
        df.fillna(na_value, inplace=True)
        return tf.data.Dataset.from_tensor_slices(tuple(df[col] for col in df.columns))

    def load_csv(
        self,
        filename: str,
        sep: str,
        in_memory: bool,
        column_dtypes: List[str],
        na_value: str,
    ) -> Tuple[tf.data.Dataset, Optional[int]]:
        if in_memory:
            df = pd.read_csv(filename, sep=sep, quoting=3)
            for dtype, col in zip(column_dtypes, df.columns):
                df[col] = df[col].astype(dtype)
            return self.dataframe_to_dataset(df, na_value), df.shape[0]
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
