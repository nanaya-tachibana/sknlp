import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate

from sknlp.data import ClassificationDataset
from sknlp.metrics import PrecisionWithLogits, RecallWithLogits
from sknlp.callbacks import FScore
from sknlp.module.base_model import SupervisedNLPModel


class DeepClassifier(SupervisedNLPModel):

    def __init__(self, classes, is_multilabel=True, segmenter='jieba',
                 embed_size=100, max_length=None, vocab=None, token2vec=None,
                 **kwargs):
        super().__init__(classes,
                         segmenter=segmenter,
                         embed_size=embed_size,
                         max_length=max_length,
                         vocab=vocab,
                         token2vec=token2vec, **kwargs)
        self._is_multilabel = is_multilabel

    def get_loss(self):
        if self._is_multilabel:
            return tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            return tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def get_callbacks(self):
        return [FScore()]

    def get_metrics(self):
        if self._is_multilabel:
            return [PrecisionWithLogits(), RecallWithLogits()]
        else:
            return [PrecisionWithLogits(logits2scores='softmax'),
                    RecallWithLogits(logits2scores='softmax')]

    def dataset_transform(
        self, dataset, vocab, labels, max_length, segmenter,
        dataset_size=-1, batch_size=32, shuffle=True
    ):
        if not hasattr(self, '_dataset_transformer'):
            self._dataset_transformer = ClassificationDataset(
                vocab, labels, max_length=max_length, text_segmenter=segmenter
            )
        return self._dataset_transformer.transform_and_batchify(
            dataset, batch_size, shuffle=shuffle,
            shuffle_buffer_size=dataset_size
        )

    def dataset_batchify(
        self, dataset, vocab, labels, batch_size=32, shuffle=True
    ):
        d = ClassificationDataset(vocab, labels)
        return d.batchify(dataset, batch_size, shuffle=shuffle)

    def get_config(self):
        return {
            **super().get_config(),
            'is_multilabel': self._is_multilabel,
            'task': 'classification'
        }

    def get_custom_objects(self):
        return {**super().get_custom_objects(),
                'PrecisionWithLogits': PrecisionWithLogits,
                'RecallWithLogits': RecallWithLogits}

    def score_func(self, y, prediction_scores):
        precision, recall, f_score, num = [], [], [], []
        names = list(self._class2idx.keys())
        if not self._is_multilabel:
            predictions = np.argmax(prediction_scores, axis=1).tolist()
            y = np.argmax(y, axis=1).tolist()
        else:
            predictions = prediction_scores > 0.5

        p, r, f, n = precision_recall_fscore_support(y, predictions)
        precision.extend(p.tolist())
        recall.extend(r.tolist())
        f_score.extend(f.tolist())
        num.extend(n.tolist())

        # if y.ndim > 1 and y.shape[1] == 2 and not self._is_multilabel:
        #     p, r, f, n = precision_recall_fscore_support(y, predictions,
        #                                                  average='binary')
        # else:
        p, r, f, n = precision_recall_fscore_support(y, predictions, average='micro')
        precision.append(p)
        recall.append(r)
        f_score.append(f)
        num.append(n)
        names.append('avg')
        return {'names': names,
                'num': num,
                'precision': precision,
                'recall': recall,
                'f_score': f_score}

    def format_score(self, score):
        names = score['names']
        num = score['num']
        precision = score['precision']
        recall = score['recall']
        f_score = score['f_score']
        return tabulate(
            {
                'Class Name(Num)': [
                    '%s' % c if n is None else '%s(%d)' % (c, n)
                    for c, n in zip(names, num)
                ],
                'f(p, r)': [
                    '%.2f(%.2f, %.2f)' % (f * 100, p * 100, r * 100)
                    for (p, r, f) in zip(precision, recall, f_score)
                ]
            },
            headers='keys',
            tablefmt='github'
        )

    # def build_output_layer(self, inputs):
    #     logits = super().build_output_layer(inputs)
    #     if self._is_multilabel:
    #         return tf.keras.activations.sigmoid(logits)
    #     else:
    #         return tf.keras.activations.softmax(logits, axis=-1)
