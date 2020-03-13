# -*- coding: utf-8 -*-
import pandas as pd
import tensorflow as tf
from .text_segmenter import get_segmenter


class NLPDataset:

    def __init__(self,
                 text_segmenter='list',
                 text_dtype=tf.string,
                 label_dtype=tf.string,
                 text_padding_shape=(None,),
                 label_padding_shape=(),
                 text_padding_value='<pad>',
                 label_padding_value=''):
        self.text_cutter = get_segmenter(text_segmenter)
        self.text_dtype = text_dtype
        self.label_dtype = label_dtype
        self.text_padding_shape = text_padding_shape
        self.label_padding_shape = label_padding_shape
        self.text_padding_value = text_padding_value
        self.label_padding_value = label_padding_value

    def text_transform(self, text):
        return self.text_cutter(text.numpy().decode('utf-8'))

    def label_transform(self, label):
        return label.numpy().decode('utf-8')

    def transform(self, dataset):

        def func(text, label):
            return self.text_transform(text), self.label_transform(label)

        return dataset.map(
            lambda t, l: tf.py_function(
                func, inp=[t, l], Tout=(self.text_dtype, self.label_dtype)
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    def batchify(self, dataset, batch_size,
                 shuffle=True, shuffle_buffer_size=100000):
        if shuffle:
            if shuffle_buffer_size <= 0:
                shuffle_buffer_size = 100000
            dataset = dataset.shuffle(shuffle_buffer_size)

        return (
            dataset
            .padded_batch(
                batch_size,
                (self.text_padding_shape, self.label_padding_shape),
                padding_values=(self.text_padding_value,
                                self.label_padding_value)
            )
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

    def transform_and_batchify(self, dataset, batch_size,
                               shuffle=True, shuffle_buffer_size=100000):
        return self.batchify(self.transform(dataset),
                             batch_size,
                             shuffle=shuffle,
                             shuffle_buffer_size=shuffle_buffer_size)

    @classmethod
    def dataframe_to_dataset(cls, df):
        df.columns = ['text', 'label']
        df.fillna('', inplace=True)
        return tf.data.Dataset.from_tensor_slices((df.text, df.label))

    @classmethod
    def load_csv(cls, filename, sep, in_memory):
        if in_memory:
            df = pd.read_csv(filename, sep=sep, dtype=str)
            return cls.dataframe_to_dataset(df), df.shape[0]
        return (
            tf.data.experimental.CsvDataset(filename,
                                            (tf.string, tf.string),
                                            header=True,
                                            field_delim=sep,
                                            na_value=''),
            -1
        )


# def _supervised_dataset_transform(dataset,
#                                   text_transform_func,
#                                   label_transform_func,
#                                   text_dtype=tf.string,
#                                   label_dtype=tf.string,
#                                   dataset_size=-1,
#                                   batch_size=32,
#                                   text_padding_shape=(None,),
#                                   label_padding_shape=(),
#                                   text_padding_value='<pad>',
#                                   label_padding_value='',
#                                   shuffle=True,
#                                   shuffle_buffer_size=100000):
#     def _func(text, label):
#         return (

#         )

#     if shuffle:
#         if dataset_size != -1:
#             shuffle_buffer_size = dataset_size
#         dataset = dataset.shuffle(shuffle_buffer_size)

#     dataset = dataset.map(
#         lambda t, l: tf.py_function(
#             _func, inp=[t, l], Tout=(text_dtype, label_dtype)
#         ),
#         num_parallel_calls=tf.data.experimental.AUTOTUNE
#     )
#     if batch_size is not None:
#         return (
#             dataset
#             .padded_batch(
#                 batch_size, (text_padding_shape, label_padding_shape),
#                 padding_values=(text_padding_value, label_padding_value)
#             ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#         )
#     return dataset





# def _make_dataset_from_csv(filename,
#                            text_transform_func,
#                            label_transform_func,
#                            text_dtype=tf.string,
#                            label_dtype=tf.string,
#                            sep='\t',
#                            in_memory=True,
#                            batch_size=32,
#                            text_padding_shape=(None,),
#                            label_padding_shape=(),
#                            text_padding_value='<pad>',
#                            label_padding_value='',
#                            shuffle=True,
#                            shuffle_buffer_size=100000):
#     dataset, size = _get_csv_dataset(filename, sep, in_memory)
#     return _supervised_dataset_transform(
#         dataset, text_transform_func, label_transform_func,
#         text_dtype=text_dtype, label_dtype=label_dtype,
#         dataset_size=size, batch_size=batch_size,
#         text_padding_shape=text_padding_shape,
#         label_padding_shape=label_padding_shape,
#         text_padding_value=text_padding_value,
#         label_padding_value=label_padding_value,
#         shuffle=shuffle, shuffle_buffer_size=shuffle_buffer_size
#     )
