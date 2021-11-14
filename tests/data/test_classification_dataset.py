# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf

from sknlp.vocab import Vocab
from sknlp.data.classification_dataset import ClassificationDataset


def test_label_binarizer(text_with_empty):
    vocab = Vocab(["x", "y"])
    d = ClassificationDataset(vocab, ["a", "c", "b"], csv_file=text_with_empty)
    np.testing.assert_array_equal(d.py_label_binarizer(["c", "a", "d"]), [1, 1, 0])


def test_transform_func(text_with_empty):
    vocab = Vocab(["x", "y"])
    d = ClassificationDataset(
        vocab,
        ["1", "2"],
        segmenter="char",
        is_multilabel=True,
        csv_file=text_with_empty,
        max_length=2,
    )
    data = d.py_transform(tf.constant("xz"), tf.constant("1|2"))
    np.testing.assert_array_equal(data[0], [vocab["x"], vocab[vocab.unk]])
    np.testing.assert_array_equal(data[1], [1, 1])


def test_create_from_csv(text_without_empty):
    vocab = Vocab(["你", "啊", "拿", "好", "我"])
    labels = ["1", "2"]
    for in_memory in (True, False):
        d = ClassificationDataset(
            vocab,
            labels,
            is_multilabel=True,
            csv_file=text_without_empty,
            in_memory=in_memory,
        )
        dataset = d.batchify(2, shuffle=False)
        for text, label in dataset:
            text = text.numpy()
            assert text[1][-1] == vocab[vocab.pad]
            assert text[1][0] == vocab["我"]
            label = label.numpy()
            np.testing.assert_array_equal(label, [[1, 0], [1, 1]])


def test_classification_dataset_transform():
    vocab = Vocab(["x", "y"])
    df = pd.DataFrame({"text": ["xxx", "yyyyy"], "label": ["1|2", "2"]})
    d = ClassificationDataset(
        vocab, ["2", "1"], segmenter="char", is_multilabel=True, X=df.text, y=df.label
    )
    dataset = d.batchify(2, shuffle=False)
    for text, label in dataset:
        text = text.numpy()
        assert text[0][-1] == vocab[vocab.pad]
        assert text[1][0] == vocab["y"]
        label = label.numpy()
        np.testing.assert_array_equal(label, [[1, 1], [1, 0]])