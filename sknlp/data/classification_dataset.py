# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from .nlp_dataset import NLPDataset
from .utils import serialize_example


class ClassificationDataset(NLPDataset):

    def __init__(self,
                 vocab,
                 labels,
                 is_multilabel=True,
                 max_length=80,
                 text_segmenter='list'):
        super().__init__(text_segmenter=text_segmenter,
                         text_dtype=tf.int32,
                         label_dtype=tf.float32,
                         text_padding_shape=(None,),
                         label_padding_shape=(None,),
                         text_padding_value=vocab[vocab.pad],
                         label_padding_value=0.0)
        self.vocab = vocab
        self.labels = labels
        self.is_multilabel = is_multilabel
        self.max_length = max_length
        self.label2idx = dict(zip(labels, range(len(labels))))

    def text_transform(self, text):
        tokens = super().text_transform(text)
        return np.array(
            [self.vocab[t] for t in tokens[:self.max_length]], dtype=np.int32
        )

    def label_binarizer(self, labels):
        label2idx = self.label2idx
        res = np.zeros(len(label2idx), dtype=np.float32)
        res[[label2idx[l] for l in labels if l in label2idx]] = 1
        return res

    def label_transform(self, label):
        label = super().label_transform(label)
        if self.is_multilabel:
            labels = label.split('|')
        else:
            labels = [label]
        return self.label_binarizer(labels)

    def to_tfrecord(self, filename, dataset):

        def func(text, label):
            return tf.reshape(
                serialize_example(
                    (self.text_transform(text), self.label_transform(label)),
                    ('tensor', 'tensor')
                ), ()
            )

        tf_writer = tf.data.experimental.TFRecordWriter(filename)
        tf_writer.write(dataset.map(
            lambda t, l: tf.py_function(func, inp=[t, l], Tout=tf.string),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ))

    def parse_tfrecord(self, dataset):

        def func(record):
            parsed_record = tf.io.parse_single_example(
                record,
                {
                    'feature0': tf.io.FixedLenFeature(
                        [], tf.string, default_value=''
                    ),
                    'feature1': tf.io.FixedLenFeature(
                        [], tf.string, default_value=''
                    )
                }
            )
            return (
                tf.io.parse_tensor(parsed_record['feature0'], tf.int32),
                tf.io.parse_tensor(parsed_record['feature1'], tf.float32)
            )

        return dataset.map(func)
