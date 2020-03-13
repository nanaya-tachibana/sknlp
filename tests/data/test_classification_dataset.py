# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np
import tensorflow as tf

from sknlp.vocab import Vocab
from sknlp.data.classification_dataset import ClassificationDataset


def test_label_binarizer():
    vocab = Vocab(counter=Counter({'x': 10, 'y': 2}))
    d = ClassificationDataset(vocab, ['a', 'c', 'b'])
    np.testing.assert_array_equal(d.label_binarizer(['c', 'a', 'd']),
                                  [1, 1, 0])


def test_transform_func():
    vocab = Vocab(counter=Counter({'x': 10, 'y': 2}))
    d = ClassificationDataset(vocab, ['1', '2'], max_length=2)
    assert d.text_transform(tf.constant('xz')) == [4, vocab[vocab.unk]]
    np.testing.assert_array_equal(d.label_transform(tf.constant('1|2')),
                                  [1, 1])


def test_create_from_csv(tmp_path):
    vocab = Vocab(Counter({'你': 1, '啊': 2, '拿': 3, '好': 4, '我': 5}))
    test_file = tmp_path / 'test.txt'
    test_file.write_text('text\tlabel\n你啊拿好\t1\n我好\t2|1\n')
    labels = ['1', '2']
    d = ClassificationDataset(vocab, labels)
    for in_memory in (True, False):
        dataset, size = ClassificationDataset.load_csv(str(test_file),
                                                       sep='\t',
                                                       in_memory=in_memory)
        dataset = d.transform_and_batchify(dataset, 2,
                                           shuffle=False,
                                           shuffle_buffer_size=size)
        for text, label in dataset:
            text = text.numpy()
            assert text[1][-1] == vocab[vocab.pad]
            assert text[1][0] == 4
            label = label.numpy()
            np.testing.assert_array_equal(label, [[1, 0], [1, 1]])


def test_classification_dataset_transform():
    vocab = Vocab(counter=Counter({'x': 10, 'y': 2}))
    d = ClassificationDataset(vocab, ['2', '1'])
    dataset = tf.data.Dataset.from_tensor_slices((['xxx', 'yyyyy'],
                                                  ['1|2', '2']))
    dataset = d.transform_and_batchify(dataset, 2, shuffle=False)
    for text, label in dataset:
        text = text.numpy()
        assert text[0][-1] == vocab[vocab.pad]
        assert text[1][0] == 5
        label = label.numpy()
        np.testing.assert_array_equal(label, [[1, 1], [1, 0]])
