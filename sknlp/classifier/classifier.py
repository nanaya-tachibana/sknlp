import functools
import json
import os
import shutil
import tempfile
import logging

import mxnet as mx
from mxnet.gluon import nn
import gluonnlp
import numpy as np

from ..base import DeepSupervisedModel
from ..data import ClassifyDataset, InMemoryDataset
from ..data.batchify import Pad, Stack
from ..embedding import Token2vec
from ..encode import TextCNN, TextRCNN, TextRNN
from ..segmenter import Segmenter
from ..metric import classify_f_score
from ..utils.array import sequence_mask
from ..utils.file import make_tarball

from .utils import logits2classes, logits2scores, scores2classes


logger = logging.getLogger(__name__)


def batchify(padding, one_batch):
    (inputs, length), labels = gluonnlp.data.batchify.Tuple(
        Pad(axis=0, pad_val=padding, ret_length=True), Stack()
    )(one_batch)
    inputs = inputs.transpose((1, 0))
    mask = sequence_mask(np.ones_like(inputs), length.astype('int'))
    return inputs, mask, labels.astype('float32')


class DeepClassifier(DeepSupervisedModel):

    def __init__(
        self, num_classes, encode_layer, embedding_layer=None,
        vocab=None, is_multilabel=False, label2idx=None,
        segmenter='jieba', max_length=100, embed_size=100, **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer
        self.encode_layer = encode_layer
        self._vocab = vocab
        self._num_classes = num_classes
        self._is_multilabel = is_multilabel
        self._segmenter = segmenter
        self._cut = Segmenter(segmenter).cut
        self._max_length = max_length
        self._embed_size = embed_size
        self._label2idx = label2idx

        self.meta = {
            'num_classes': num_classes,
            'is_multilabel': is_multilabel,
            'label2idx': label2idx,
            'max_length': max_length,
            'segmenter': segmenter,
            'embed_size': embed_size,
        }

    def _build(self, ctx, initialize=True):
        if self.embedding_layer is None:
            self.embedding_layer = Token2vec(
                self._vocab, self._embed_size, loss=None
            )
        self._trainable = {
            'embedding': self.embedding_layer,
            'encode': self.encode_layer
        }
        if self._is_multilabel:
            self.loss = mx.gluon.loss.SigmoidBCELoss()
        else:
            self.loss = mx.gluon.loss.SoftmaxCELoss(sparse_label=False)

        if initialize:
            self.embedding_layer._build(ctx, initialize=initialize)
            self.encode_layer.initialize(init=mx.init.Xavier(), ctx=ctx)
            self.loss.initialize(init=mx.init.Xavier(), ctx=ctx)
        self.encode_layer.hybridize(static_alloc=True)
        self.loss.hybridize(static_alloc=True)

    def _get_or_build_dataset(self, dataset, X, y):
        assert (X and y) or dataset
        if dataset:
            if not hasattr(self, 'idx2labels'):
                self.idx2labels = dataset.idx2labels
            return dataset
        dataset = ClassifyDataset(
            InMemoryDataset(X, y),
            vocab=self._vocab, label2idx=self._label2idx,
            segmenter=self._cut, max_length=self._max_length
        )
        if not hasattr(self, 'idx2labels'):
            self.idx2labels = dataset.idx2labels
        return dataset

    def _valid_log(self, valid_dataset):
        score = self.score(dataset=valid_dataset)
        logger.info(self.format_f_score(score))
        return score

    def format_f_score(self, score):
        score_strings = []
        for key in score:
            format_string = '%s(%d) %.2f(%.2f, %.2f)'
            if key != 'avg' and score[key][-1] > 0:
                score_strings.append(
                    format_string % (
                        key, score[key][3],
                        score[key][2] * 100,  # F score
                        score[key][0] * 100,  # precision
                        score[key][1] * 100  # recall
                    )
                )
        score_strings.append(
            'avg %.2f(%.2f, %.2f)' % (
                score['avg'][2] * 100,  # F score
                score['avg'][0] * 100,  # precision
                score['avg'][1] * 100  # recall
            )
        )
        return '\n'.join(score_strings)

    def _decode(self, logits, threshold, return_score=False):
        threshold = np.array([
            threshold.get(l, 0.5)
            for l in self.idx2labels(range(self._num_classes))
        ])
        if return_score:
            scores = logits2scores(logits, self._is_multilabel)
            classes = scores2classes(scores, self._is_multilabel, threshold)
            class_scores = []
            for i, c in enumerate(classes):
                if isinstance(c, list):
                    class_scores.append([scores[i, j] for j in c])
                else:
                    class_scores.append(scores[i, c])
            return classes, class_scores
        return logits2classes(logits, self._is_multilabel, threshold=threshold)

    def _calculate_logits(self, input, mask, *args):
        return self.encode_layer(self.embedding_layer(input), mask)

    def _calculate_loss(self, inputs, mask, labels):
        logits = self._calculate_logits(inputs, mask)
        return self.loss(logits, labels), None

    def _batchify_fn(self):
        vocab = self._vocab
        return functools.partial(batchify, vocab[vocab.padding_token])

    def predict(
        self, X=None, dataset=None, threshold=None,
        batch_size=512, return_score=False
    ):
        assert self._trained
        assert dataset or X
        _threshold = threshold or dict()

        if dataset is None:
            dataset = self._get_or_build_dataset(dataset, X, ['O'] * len(X))
        dataloader = self._build_dataloader(dataset, batch_size, False, 'keep')

        predictions = []
        for one_batch in dataloader:
            logits = self._forward(self._calculate_logits, one_batch)
            if not return_score:
                predictions.extend(self._decode(logits.asnumpy(), _threshold))
            else:
                classes, scores = self._decode(
                    logits.asnumpy(), _threshold, return_score=True
                )
                predictions.extend(classes)
        dataloader.reset()

        if self._is_multilabel:
            orignal_labels = [self.idx2labels(p) for p in predictions]
        else:
            orignal_labels = self.idx2labels(predictions)
        if return_score:
            return orignal_labels, scores
        else:
            return orignal_labels

    def score(
        self, X=None, y=None, dataset=None, threshold=None, batch_size=512
    ):
        assert self._trained
        assert dataset or X

        dataset = self._get_or_build_dataset(dataset, X, y)
        predictions = self.predict(
            dataset=dataset, threshold=threshold, batch_size=batch_size
        )
        _y = np.vstack([l for _, l in dataset])
        if self._is_multilabel:
            _predictions = dataset._binarizer.transform(predictions)
        else:
            _y = _y.argmax(axis=1)
            _predictions = dataset._binarizer.transform([
                [p] for p in predictions
            ]).argmax(axis=1)
        return classify_f_score(
            _y, _predictions, self._is_multilabel,
            labels=self.idx2labels(range(self._num_classes))
        )

    def save(self, file_path: str) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.embedding_layer.save(os.path.join(temp_dir, 'embedding'))
            self.encode_layer.export(os.path.join(temp_dir, 'encode'))
            with open(os.path.join(temp_dir, 'meta.json'), 'w') as f:
                f.write(json.dumps(self.meta, ensure_ascii=False))
            make_tarball(file_path, temp_dir)

    @classmethod
    def _load_embedding_layer(cls, file_path, update, ctx):
        return Token2vec.load(file_path, update=update, ctx=ctx)

    @classmethod
    def _load(cls, temp_dir, meta, inputs=None, update=False, ctx=mx.cpu()):
        embedding_layer = cls._load_embedding_layer(
            os.path.join(temp_dir, 'embedding.tar'), update, ctx
        )
        if inputs is None:
            inputs = ['data0', 'data1']
        encode_layer = nn.SymbolBlock.imports(
            os.path.join(temp_dir, 'encode-symbol.json'), inputs,
            os.path.join(temp_dir, 'encode-0000.params'), ctx=ctx
        )
        for name, param in encode_layer.collect_params().items():
            param.grad_req = 'null'
        ins = cls(
            meta['num_classes'], encode_layer,
            embedding_layer=embedding_layer,
            vocab=embedding_layer._vocab,
            is_multilabel=meta['is_multilabel'],
            label2idx=meta['label2idx'], segmenter=meta['segmenter'],
            max_length=meta['max_length'], embed_size=meta['embed_size'],
            ctx=ctx
        )
        ins.meta = meta
        ins._trained = True
        ins._build(ctx, initialize=False)
        return ins

    @staticmethod
    def load(file_path, update=False, ctx=mx.cpu()):
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(file_path, temp_dir, 'tar')
            with open(os.path.join(temp_dir, 'meta.json')) as f:
                meta = json.loads(f.read())

            if meta['model_type'] == 'builtin-text_cnn_classifier':
                return TextCNNClassifier._load(
                    temp_dir, meta, update=update, ctx=ctx
                )
            elif meta['model_type'] == 'builtin-text_rnn_classifier':
                return TextRNNClassifier._load(
                    temp_dir, meta, update=update, ctx=ctx
                )
            elif meta['model_type'] == 'builtin-text_rcnn_classifier':
                return TextRCNNClassifier._load(
                    temp_dir, meta, update=update, ctx=ctx
                )
            else:
                raise ValueError('unknown model type.')


def cnn_batchify(padding, min_length, one_batch):
    (inputs, length), labels = gluonnlp.data.batchify.Tuple(
        Pad(axis=0, pad_val=padding, ret_length=True, min_length=min_length),
        Stack()
    )(one_batch)
    inputs = inputs.transpose((1, 0))
    mask = sequence_mask(np.ones_like(inputs), length.astype('int'))
    return inputs, mask, labels.astype('float32')


class TextCNNClassifier(DeepClassifier):

    def __init__(
        self, num_classes, encode_layer=None, embedding_layer=None,
        vocab=None, is_multilabel=False, label2idx=None, segmenter='jieba',
        max_length=100, embed_size=100,
        num_filters=(25, 50, 75, 100), ngram_filter_sizes=(1, 2, 3, 4),
        conv_layer_activation='tanh', num_highways=1, dropout=0,
        num_fc_layers=2, fc_hidden_size=512, fc_activation='tanh',
        ctx=mx.cpu(), **kwargs
    ):
        if encode_layer is None:
            encode_layer = TextCNN(
                embed_size=embed_size,
                num_filters=num_filters,
                ngram_filter_sizes=ngram_filter_sizes,
                conv_layer_activation=conv_layer_activation,
                num_highways=num_highways,
                dropout=dropout,
                num_fc_layers=num_fc_layers,
                fc_hidden_size=fc_hidden_size,
                output_size=num_classes,
                prefix='encode_'
            )
        super().__init__(
            num_classes, encode_layer, embedding_layer=embedding_layer,
            is_multilabel=is_multilabel, label2idx=label2idx,
            vocab=vocab, segmenter=segmenter, max_length=max_length,
            embed_size=embed_size, ctx=ctx, **kwargs
        )
        self.meta.update({
            'num_filters': list(num_filters),
            'ngram_filter_sizes': list(ngram_filter_sizes),
            'conv_layer_activation': conv_layer_activation,
            'num_highways': num_highways,
            'num_fc_layers': num_fc_layers,
            'fc_hidden_size': fc_hidden_size,
            'fc_activation': fc_activation
        })
        self.meta.update({'model_type': 'builtin-text_cnn_classifier'})

    def _batchify_fn(self):
        vocab = self._vocab
        return functools.partial(
            cnn_batchify, vocab[vocab.padding_token],
            max(self.meta['ngram_filter_sizes'])
        )


class TextRNNClassifier(DeepClassifier):

    def __init__(
        self, num_classes, encode_layer=None, embedding_layer=None,
        is_multilabel=False, label2idx=None, vocab=None, segmenter='jieba',
        max_length=100, embed_size=100,
        num_rnn_layers=1, projection_size=128, hidden_size=1024,
        cell_clip=3, projection_clip=3, dropout=0.5, dense_connection='last',
        num_fc_layers=2, fc_hidden_size=512, fc_activation='tanh',
        ctx=mx.cpu(), **kwargs
    ):
        if encode_layer is None:
            encode_layer = TextRNN(
                num_rnn_layers=num_rnn_layers,
                projection_size=projection_size,
                hidden_size=hidden_size,
                cell_clip=cell_clip,
                projection_clip=projection_clip,
                fc_activation=fc_activation,
                num_fc_layers=num_fc_layers,
                fc_hidden_size=fc_hidden_size,
                dropout=dropout,
                dense_connection=dense_connection,
                output_size=num_classes,
                prefix='encode_'
            )
        super().__init__(
            num_classes, encode_layer, embedding_layer=embedding_layer,
            is_multilabel=is_multilabel, label2idx=label2idx,
            vocab=vocab, segmenter=segmenter, max_length=max_length,
            embed_size=embed_size, ctx=ctx, **kwargs
        )
        self.meta.update({
            'num_rnn_layers': num_rnn_layers,
            'projection_size': projection_size,
            'hidden_size': hidden_size,
            'cell_clip': cell_clip,
            'projection_clip': projection_clip,
            'dropout': dropout,
            'dense_connection': dense_connection,
            'num_fc_layers': num_fc_layers,
            'fc_hidden_size': fc_hidden_size,
            'fc_activation': fc_activation
        })
        self.meta.update({'model_type': 'builtin-text_rnn_classifier'})


class TextRCNNClassifier(DeepClassifier):

    def __init__(
        self, num_classes, encode_layer=None, embedding_layer=None,
        is_multilabel=False, label2idx=None, vocab=None, segmenter='jieba',
        max_length=100, embed_size=100,
        num_rnn_layers=1, projection_size=128, hidden_size=1024,
        cell_clip=3, projection_clip=3, dropout=0.5, kmax=2,
        num_fc_layers=2, fc_hidden_size=512, fc_activation='tanh',
        ctx=mx.cpu(), **kwargs
    ):
        if encode_layer is None:
            encode_layer = TextRCNN(
                num_rnn_layers=num_rnn_layers,
                projection_size=projection_size,
                hidden_size=hidden_size,
                cell_clip=cell_clip,
                projection_clip=projection_clip,
                kmax=kmax,
                num_fc_layers=num_fc_layers,
                fc_hidden_size=fc_hidden_size,
                fc_activation=fc_activation,
                dropout=dropout,
                output_size=num_classes,
                prefix='encode_'
            )
        super().__init__(
            num_classes, encode_layer, embedding_layer=embedding_layer,
            is_multilabel=is_multilabel, label2idx=label2idx,
            vocab=vocab, segmenter=segmenter, max_length=max_length,
            embed_size=embed_size, ctx=ctx, **kwargs
        )
        self.meta.update({
            'num_rnn_layers': num_rnn_layers,
            'projection_size': projection_size,
            'hidden_size': hidden_size,
            'cell_clip': cell_clip,
            'projection_clip': projection_clip,
            'kmax': kmax,
            'dropout': dropout,
            'num_fc_layers': num_fc_layers,
            'fc_hidden_size': fc_hidden_size,
            'fc_activation': fc_activation
        })
        self.meta.update({'model_type': 'builtin-text_rcnn_classifier'})
