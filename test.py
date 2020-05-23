import os
import itertools
import jieba_fast as jieba
import pandas as pd
import tempfile

import tensorflow as tf
from sknlp.module import Token2vec
from sknlp.data import ClassificationDataset
from sknlp.module.classifiers import TextRNNClassifier

jieba.set_dictionary('dict_small.txt')

if __name__ == '__main__':
    tdf = pd.read_csv('train.csv', sep='\t', dtype=str, usecols=['text', 'label'])
    tdf.fillna('', inplace=True)
    vdf = pd.read_csv('test.csv', sep='\t', dtype=str, usecols=['text', 'label'])
    vdf.fillna('', inplace=True)
    labels = sorted(list(set(itertools.chain(
        *tdf.label.apply(lambda x: x.split('|') if x != '' else []))))
    )
    print(len(labels), labels)

    tv = Token2vec.load('new_jieba.tar')
    # dataset = ClassificationDataset(tv.vocab, labels)
    # dataset.to_tfrecord('tfrecord', ClassificationDataset.dataframe_to_dataset(tdf))
    # dataset.to_tfrecord('tfrecord_valid', ClassificationDataset.dataframe_to_dataset(vdf))
    # # d = dataset.parse_tfrecord(tf.data.TFRecordDataset('tfrecord'))
    tv.freeze()
    clf = TextRNNClassifier(labels,
                            is_multilabel=False,
                            rnn_projection_size=200,
                            rnn_recurrent_clip=3,
                            rnn_projection_clip=3,
                            rnn_input_dropout=0.5,
                            rnn_recurrent_dropout=0.5,
                            token2vec=tv)
    clf.fit(
        # dataset=dataset.parse_tfrecord(tf.data.TFRecordDataset('tfrecord')),
        # valid_dataset=dataset.parse_tfrecord(tf.data.TFRecordDataset('tfrecord_valid')),
        X=tdf.text, y=tdf.label,
        valid_X=vdf.text, valid_y=vdf.label,
        n_epochs=2, batch_size=128, verbose=1
    )
    score = clf.score(vdf.text, vdf.label)
    print(clf.format_score(score))
