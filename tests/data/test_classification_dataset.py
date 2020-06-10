# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf

from sknlp.vocab import Vocab
from sknlp.data.classification_dataset import ClassificationDataset


def test_label_binarizer(text_with_empty):
    vocab = Vocab(counter=Counter({'x': 10, 'y': 2}))
    d = ClassificationDataset(vocab, ['a', 'c', 'b'], csv_file=text_with_empty)
    np.testing.assert_array_equal(d._label_binarizer(['c', 'a', 'd']), [1, 1, 0])


def test_transform_func(text_with_empty):
    vocab = Vocab(counter=Counter({'x': 10, 'y': 2}))
    d = ClassificationDataset(vocab, ['1', '2'], csv_file=text_with_empty, max_length=2)
    np.testing.assert_array_equal(
        d._text_transform(tf.constant('xz')), [4, vocab[vocab.unk]]
    )
    np.testing.assert_array_equal(d._label_transform(tf.constant('1|2')), [1, 1])


def test_create_from_csv(text_without_empty):
    vocab = Vocab(Counter({'你': 1, '啊': 2, '拿': 3, '好': 4, '我': 5}))
    labels = ['1', '2']
    for in_memory in (True, False):
        d = ClassificationDataset(
            vocab, labels, csv_file=text_without_empty, in_memory=in_memory
        )
        dataset = d.batchify(2, shuffle=False)
        for text, label in dataset:
            text = text.numpy()
            assert text[1][-1] == vocab[vocab.pad]
            assert text[1][0] == 4
            label = label.numpy()
            np.testing.assert_array_equal(label, [[1, 0], [1, 1]])


def test_classification_dataset_transform():
    vocab = Vocab(counter=Counter({'x': 10, 'y': 2}))
    df = pd.DataFrame({
        'text': ['xxx', 'yyyyy'], 'label': ['1|2', '2']
    })
    d = ClassificationDataset(vocab, ['2', '1'], df=df)
    dataset = d.batchify(2, shuffle=False)
    for text, label in dataset:
        text = text.numpy()
        assert text[0][-1] == vocab[vocab.pad]
        assert text[1][0] == 5
        label = label.numpy()
        np.testing.assert_array_equal(label, [[1, 1], [1, 0]])
