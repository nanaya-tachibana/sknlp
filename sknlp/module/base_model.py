import json
import os
import itertools
from collections import Counter
import logging

import tensorflow as tf
import numpy as np
import pandas as pd

from sknlp.data import NLPDataset
from sknlp.data.text_segmenter import get_segmenter
from sknlp.vocab import Vocab
from sknlp.metrics import logits2scores, scores2classes
from .embedding import Token2vec

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
logger.addHandler(stream)


class BaseNLPModel:

    def __init__(self, segmenter="jieba", embed_size=100,
                 max_length=None, vocab=None, token2vec=None):
        self._max_length = max_length
        self._token2vec = token2vec
        if token2vec is not None:
            self._segmenter = token2vec.segmenter
            self._vocab = token2vec.vocab
            self._embed_size = token2vec.embed_size
        else:
            self._vocab = vocab
            self._segmenter = segmenter
            self._embed_size = embed_size
        self._built = False

    @staticmethod
    def build_vocab(texts, segment_func, min_frequency=5):
        counter = Counter(
            itertools.chain.from_iterable(segment_func(text) for text in texts)
        )
        return Vocab(counter, min_frequency=min_frequency)

    def build_token2vec(self, vocab):
        self._vocab = vocab
        self._token2vec = Token2vec(vocab,
                                    self._embed_size,
                                    self._segmenter)

    def build_encode_layer(self, inputs):
        raise NotImplementedError()

    def build_output_layer(self, inputs):
        raise NotImplementedError()

    def build(self):
        if self._token2vec is None:
            self.build_token2vec(self._vocab)
        inputs = self._token2vec.output
        self._model = tf.keras.Model(
            inputs=self._token2vec.input,
            outputs=self.build_output_layer(self.build_encode_layer(inputs))
        )
        self._built = True

    def get_loss(self):
        raise NotImplementedError()

    def get_metrics(self):
        raise NotImplementedError()

    def get_callbacks(self):
        raise NotImplementedError()

    def get_custom_objects(self):
        return {}

    def save_vocab(self, filepath, filename="vocab.json"):
        with open(os.path.join(filepath, filename), "w") as f:
            f.write(self._vocab.to_json())

    def save_config(self, filepath, filename="meta.json"):
        with open(os.path.join(filepath, filename), "w") as f:
            f.write(json.dumps(self.get_config()))

    def save(self, filepath):
        self._model.save(os.path.join(filepath, "model"), save_format="h5")
        self.save_vocab(filepath)
        self.save_config(filepath)

    @classmethod
    def load(cls, filepath):
        with open(os.path.join(filepath, "vocab.json")) as f:
            vocab = Vocab.from_json(f.read())
        with open(os.path.join(filepath, "meta.json")) as f:
            meta = json.loads(f.read())
        meta.pop("algorithm", None)
        meta.pop("task", None)
        module = cls(vocab=vocab, **meta)
        module._model = tf.keras.models.load_model(
            os.path.join(filepath, "model"),
            custom_objects=module.get_custom_objects()
        )
        module._built = True
        return module

    def export(self, filepath, name, version="0"):
        self._model.save(os.path.join(filepath, name, version), save_format="tf")

    def get_config(self):
        return {"segmenter": self._segmenter,
                "embed_size": self._embed_size,
                "max_length": self._max_length}


class SupervisedNLPModel(BaseNLPModel):

    def __init__(self, classes, segmenter="jieba", embed_size=100,
                 max_length=None, vocab=None, token2vec=None, **kwargs):
        super().__init__(segmenter=segmenter,
                         embed_size=embed_size,
                         max_length=max_length,
                         vocab=vocab,
                         token2vec=token2vec, **kwargs)
        self._class2idx = dict(zip(list(classes), range(len(classes))))
        self._num_classes = len(classes)

    def dataset_transform(
        self, dataset, vocab, labels, max_length, segmenter,
        dataset_size=-1, batch_size=32, shuffle=True
    ):
        raise NotImplementedError()

    def dataset_batchify(
        self, dataset, vocab, labels, batch_size=32, shuffle=True
    ):
        raise NotImplementedError()

    def _get_or_create_dataset(self, X, y, dataset, batch_size, shuffle=True):
        assert not ((X is None or y is None) and dataset is None)

        if self._vocab is None:
            cut = get_segmenter(self._segmenter)
            self._vocab = self.build_vocab(X, cut)
        if dataset is not None:
            return self.dataset_batchify(
                dataset, self._vocab, self._class2idx.keys(),
                batch_size=batch_size, shuffle=shuffle
            )
        if isinstance(y[0], (list, tuple)):
            y = ["|".join(map(str, y_i)) for y_i in y]
        df = pd.DataFrame(zip(X, y))
        return self.dataset_transform(NLPDataset.dataframe_to_dataset(df),
                                      self._vocab, self._class2idx.keys(),
                                      self._max_length, self._segmenter,
                                      dataset_size=df.shape[0],
                                      batch_size=batch_size,
                                      shuffle=shuffle)

    def fit(self, X=None, y=None, *, dataset=None,
            valid_X=None, valid_y=None, valid_dataset=None,
            batch_size=128,  n_epochs=30, optimizer="adam",
            lr=1e-3, lr_update_factor=0.5, lr_update_epochs=10,
            clip=5.0, checkpoint=None, save_frequency=1, log_file="train.log",
            verbose=2):
        assert not (self._token2vec is None
                    and self._vocab is None
                    and X is None), \
            "When token2vec and vocab are both not given, X must be provided"

        self.train_dataset = self._get_or_create_dataset(
            X, y, dataset, batch_size
        )
        self.valid_dataset = self._get_or_create_dataset(
            valid_X, valid_y, valid_dataset, batch_size, shuffle=False
        )
        if not self._built:
            self.build()
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clip)
        if optimizer != "adam":
            pass
        loss = self.get_loss()
        metrics = self.get_metrics()
        callbacks = self.get_callbacks()
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        def scheduler(epoch):
            new_lr = lr * lr_update_factor ** (epoch // lr_update_epochs)
            if epoch != 0 and epoch % lr_update_epochs == 0:
                logger.info("change learning rate to %.4g" % new_lr)
            return new_lr

        lr_decay = tf.keras.callbacks.LearningRateScheduler(scheduler)
        self._model.fit(self.train_dataset, epochs=n_epochs,
                        validation_data=self.valid_dataset,
                        callbacks=[*callbacks, lr_decay,
                                   tf.keras.callbacks.CSVLogger(log_file)],
                        verbose=verbose)

    def predict(self, X=None, *, dataset=None, threshold=None, batch_size=128,
                return_scores=False):
        assert self._built
        y = None if X is None else ["O" for _ in range(len(X))]
        dataset = self._get_or_create_dataset(X, y, dataset, batch_size,
                                              shuffle=False)

        def _f(x, y):
            return x

        predictions = logits2scores(self._model.predict(dataset.map(_f)),
                                    self._is_multilabel)
        if return_scores:
            return predictions
        return scores2classes(predictions, self._is_multilabel)

    def score(self, X=None, y=None, *, dataset=None, batch_size=128):
        prediction_scores = self.predict(
            X=X, dataset=dataset, batch_size=batch_size, return_scores=True
        )
        dataset = self._get_or_create_dataset(
            X, y, dataset, batch_size, shuffle=False
        )
        y = np.vstack([yi.numpy() for _, yi in dataset])
        return self.score_func(y, prediction_scores)

    def get_config(self):
        return {**super().get_config(), "classes": list(self._class2idx.keys())}
