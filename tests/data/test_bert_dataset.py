from collections import Counter

import numpy as np
import tensorflow as tf

from sknlp.vocab import Vocab
from sknlp.data.bert_dataset import BertClassificationDataset


def test_transform_func(text_with_empty):
    vocab = Vocab(counter=Counter({'x': 10, 'y': 2}))
    d = BertClassificationDataset(vocab, ['1', '2'], csv_file=text_with_empty, max_length=10)
    np.testing.assert_array_equal(
        d._text_transform(tf.constant('xz')),
        [tf.constant('xz')]
    )
    np.testing.assert_array_equal(d._label_transform(tf.constant('1|2')), [1, 1])
