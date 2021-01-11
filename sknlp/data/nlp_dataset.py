from typing import Optional, List, Union, Tuple

import pandas as pd
import tensorflow as tf
from .text_segmenter import get_segmenter


class NLPDataset:
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        text_segmenter: str = "char",
        max_length: Optional[int] = None,
        text_dtype: tf.DType = tf.string,
        label_dtype: tf.DType = tf.string,
        text_padding_shape: tuple = (None,),
        label_padding_shape: tuple = (),
        text_padding_value: Union[int, str, float] = "<pad>",
        label_padding_value: Union[int, str, float] = "",
    ):
        assert (
            df is not None or csv_file is not None
        ), "either df or csv_file may not be None"
        if df is not None:
            self._original_dataset = self.dataframe_to_dataset(df)
            self.size = df.shape[0]
        elif csv_file is not None:
            self._original_dataset, self.size = self.load_csv(csv_file, "\t", in_memory)
        self.text_cutter = get_segmenter(text_segmenter)
        self.max_length = max_length or 99999
        self.text_dtype = text_dtype
        self.label_dtype = label_dtype
        self.text_padding_shape = text_padding_shape
        self.label_padding_shape = label_padding_shape
        self.text_padding_value = text_padding_value
        self.label_padding_value = label_padding_value

        self._dataset = self._transform(self._original_dataset)

    @property
    def text(self) -> List[str]:
        return [
            x.decode("utf-8") for x, _ in self._original_dataset.as_numpy_iterator()
        ]

    @property
    def label(self) -> List[str]:
        return [
            y.decode("utf-8") for _, y in self._original_dataset.as_numpy_iterator()
        ]

    def _text_transform(self, text: tf.Tensor) -> Union[List[str], str]:
        return self.text_cutter(text.numpy().decode("utf-8"))[: self.max_length]

    def _label_transform(self, label: tf.Tensor) -> str:
        return label.numpy().decode("utf-8")

    def _transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        def func(text, label):
            return self._text_transform(text), self._label_transform(label)

        return dataset.map(
            lambda t, l: tf.py_function(
                func, inp=[t, l], Tout=(self.text_dtype, self.label_dtype)
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
    ) -> tf.data.Dataset:
        dataset = (
            self.shuffled_dataset(shuffle_buffer_size) if shuffle else self._dataset
        )
        return dataset.padded_batch(
            batch_size,
            padded_shapes=(self.text_padding_shape, self.label_padding_shape),
            padding_values=(self.text_padding_value, self.label_padding_value),
        ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    @classmethod
    def dataframe_to_dataset(cls, df: pd.DataFrame) -> tf.data.Dataset:
        df.fillna("", inplace=True)
        return tf.data.Dataset.from_tensor_slices((df.text, df.label))

    @classmethod
    def load_csv(
        cls, filename: str, sep: str, in_memory: bool
    ) -> Tuple[tf.data.Dataset, Optional[int]]:
        if in_memory:
            df = pd.read_csv(filename, sep=sep, dtype=str)
            return cls.dataframe_to_dataset(df), df.shape[0]
        return (
            tf.data.experimental.CsvDataset(
                filename,
                (tf.string, tf.string),
                header=True,
                field_delim=sep,
                na_value="",
            ),
            None,
        )
