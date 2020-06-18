import os
import itertools
import jieba_fast as jieba
import pandas as pd

import tensorflow as tf
from sknlp.module.text2vec import Word2vec
from sknlp.module.classifiers import TextRNNClassifier

jieba.set_dictionary('data/dict_small.txt')

if __name__ == '__main__':
    tdf = pd.read_csv('data/cocs/cocs_signal_train_128.txt', sep='\t', dtype=str, usecols=['text', 'label'])
    tdf.fillna('', inplace=True)
    vdf = pd.read_csv('data/cocs/cocs_signal_test_128.txt', sep='\t', dtype=str, usecols=['text', 'label'])
    vdf.fillna('', inplace=True)
    labels = sorted(list(set(itertools.chain(
        *tdf.label.apply(lambda x: x.split('|') if x != '' else []))))
    )
    print(len(labels), labels)

    tv = Word2vec.load_archive('data/jiebax.tar')
    tv.freeze()
    clf = TextRNNClassifier(labels,
                            is_multilabel=False,
                            rnn_projection_size=200,
                            rnn_recurrent_clip=3,
                            rnn_projection_clip=3,
                            rnn_input_dropout=0.5,
                            rnn_recurrent_dropout=0.5,
                            text2vec=tv
    )
    clf.fit(
        X=tdf.text, y=tdf.label,
        n_epochs=10, batch_size=128, verbose=1
    )
    score = clf.score(vdf.text, vdf.label)
    print(clf.format_score(score))
    clf.export("data", "xyz", "0")
