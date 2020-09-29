from typing import Sequence, List, Optional

import numpy as np
import pandas as pd

import tensorflow as tf

from sknlp.vocab import Vocab

from .nlp_dataset import NLPDataset


class BertClassificationDataset(NLPDataset):

    def __init__(
        self,
        vocab: Vocab,
        labels: Sequence[str],
        df: Optional[pd.DataFrame] = None,
        csv_file: Optional[str] = None,
        in_memory: bool = True,
        is_multilabel: bool = True,
        max_length: Optional[int] = 80,
        text_segmenter: str = None
    ):
        self.vocab = vocab
        self.labels = labels
        self.is_multilabel = is_multilabel
        self.max_length = max_length or 99999
        self.label2idx = dict(zip(labels, range(len(labels))))
        super().__init__(df=df,
                         csv_file=csv_file,
                         in_memory=in_memory,
                         text_segmenter=text_segmenter,
                         text_dtype=tf.string,
                         label_dtype=tf.float32,
                         text_padding_shape=(None,),
                         label_padding_shape=(None,),
                         text_padding_value=vocab.pad,
                         label_padding_value=0.0)

    def _text_transform(self, text: tf.Tensor) -> str:
        return text.numpy().decode('utf-8')[:self.max_length]

    def _label_binarizer(self, labels: List[str]) -> np.ndarray:
        label2idx = self.label2idx
        res = np.zeros(len(label2idx), dtype=np.float32)
        res[[label2idx[label] for label in labels if label in label2idx]] = 1
        return res

    def _label_transform(self, label: tf.Tensor) -> np.ndarray:
        label = super()._label_transform(label)
        if self.is_multilabel:
            labels = label.split('|')
        else:
            labels = [label]
        return self._label_binarizer(labels)

    def batchify(
        self,
        batch_size: int,
        shuffle: bool = True,
        shuffle_buffer_size: int = 100000
    ) -> tf.data.Dataset:
        dataset = self._dataset
        if shuffle:
            if shuffle_buffer_size <= 0:
                shuffle_buffer_size = 100000
            dataset = dataset.shuffle(shuffle_buffer_size)

        return (
            dataset
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
