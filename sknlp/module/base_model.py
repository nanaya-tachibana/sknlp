from typing import Sequence, Union, Optional, Dict, Any

import json
import os
import itertools
from collections import Counter
import logging

import tensorflow as tf
import pandas as pd

from sknlp.data import NLPDataset
from sknlp.data.text_segmenter import get_segmenter
from sknlp.vocab import Vocab
from .embedding import Token2vec

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
logger.addHandler(stream)


class BaseNLPModel:

    def __init__(
        self,
        segmenter="jieba",
        embed_size=100,
        max_length=None,
        vocab=None,
        token2vec=None
    ):
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
        self._token2vec = Token2vec(vocab, self._embed_size, self._segmenter)

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

    def save_vocab(self, directory, filename="vocab.json"):
        with open(os.path.join(directory, filename), "w") as f:
            f.write(self._vocab.to_json())

    def save_config(self, directory, filename="meta.json"):
        with open(os.path.join(directory, filename), "w") as f:
            f.write(json.dumps(self.get_config()))

    def save(self, directory):
        self._model.save(os.path.join(directory, "model"), save_format="h5")
        self.save_vocab(directory)
        self.save_config(directory)

    @classmethod
    def load(cls, directory):
        with open(os.path.join(directory, "vocab.json")) as f:
            vocab = Vocab.from_json(f.read())
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.loads(f.read())
        meta.pop("algorithm", None)
        meta.pop("task", None)
        module = cls(vocab=vocab, **meta)
        module._model = tf.keras.models.load_model(
            os.path.join(directory, "model"),
            custom_objects=module.get_custom_objects()
        )
        module._built = True
        return module

    def export(self, directory, name, version="0"):
        d = os.path.join(directory, name, version)
        self._model.save(d, save_format="tf")
        self.save_vocab(d)
        self.save_config(d)

    def get_config(self):
        return {
            "segmenter": self._segmenter,
            "embed_size": self._embed_size,
            "max_length": self._max_length
        }


class SupervisedNLPModel(BaseNLPModel):

    def __init__(
        self,
        classes: Sequence[str],
        segmenter: str = "jieba",
        embed_size: int = 100,
        max_length: Optional[int] = 100,
        vocab: Optional[Vocab] = None,
        token2vec: Optional[Token2vec] = None,
        task: Optional[str] = None,
        algorithm: Optional[str] = None,
        **kwargs
    ):
        super().__init__(segmenter=segmenter,
                         embed_size=embed_size,
                         max_length=max_length,
                         vocab=vocab,
                         token2vec=token2vec,
                         **kwargs)
        self._task = task
        self._algorithm = algorithm
        self._class2idx = dict(zip(list(classes), range(len(classes))))
        self._idx2class = dict(zip(range(len(classes)), list(classes)))
        self._num_classes = len(classes)

    @classmethod
    def create_dataset_from_df(
        self,
        df: pd.DataFrame,
        vocab: Vocab,
        labels: Sequence[str]
    ) -> NLPDataset:
        raise NotImplementedError()

    def prepare_tf_dataset(
        self,
        X: Sequence[str],
        y: Sequence[str],
        dataset: NLPDataset,
        batch_size: int,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        assert not ((X is None or y is None) and dataset is None)

        if self._vocab is None:
            cut = get_segmenter(self._segmenter)
            self._vocab = self.build_vocab(X, cut)
        if dataset is not None:
            return dataset.batchify(batch_size, shuffle=shuffle)
        if isinstance(y[0], (list, tuple)):
            y = ["|".join(map(str, y_i)) for y_i in y]
        df = pd.DataFrame(zip(X, y), columns=["text", "label"])
        dataset = self.create_dataset_from_df(df, self._vocab, self._class2idx.keys())
        return dataset.batchify(
            batch_size, shuffle=shuffle, shuffle_buffer_size=df.shape[0]
        )

    def fit(
        self,
        X: Sequence[str] = None,
        y: Union[Sequence[Sequence[str]], Sequence[str]] = None,
        *,
        dataset: NLPDataset = None,
        valid_X: Sequence[str] = None,
        valid_y: Union[Sequence[Sequence[str]], Sequence[str]] = None,
        valid_dataset: NLPDataset = None,
        batch_size: int = 128,
        n_epochs: int = 30,
        optimizer: str = "adam",
        lr: float = 1e-3,
        lr_update_factor: float = 0.5,
        lr_update_epochs: int = 10,
        clip: float = 5.0,
        checkpoint: Optional[str] = None,
        save_frequency: int = 1,
        log_file: Optional[str] = None,
        verbose: int = 2
    ) -> None:
        assert not (self._token2vec is None
                    and self._vocab is None
                    and X is None), \
            "When token2vec and vocab are both not given, X must be provided"

        self.train_dataset = self.prepare_tf_dataset(X, y, dataset, batch_size)
        if ((valid_X is None or valid_y is None) and valid_dataset is None):
            self.valid_dataset = None
        else:
            self.valid_dataset = self.prepare_tf_dataset(
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
        callbacks = [*callbacks, lr_decay]
        if log_file is not None:
            callbacks.append(tf.keras.callbacks.CSVLogger(log_file))
        self._model.fit(
            self.train_dataset,
            epochs=n_epochs,
            validation_data=self.valid_dataset,
            callbacks=callbacks,
            verbose=verbose
        )

    def predict(
        self,
        X: Sequence[str] = None,
        *,
        dataset: NLPDataset = None,
        batch_size: int = 128
    ) -> tf.Tensor:
        assert self._built
        y = None if X is None else ["O" for _ in range(len(X))]
        dataset = self.prepare_tf_dataset(X, y, dataset, batch_size, shuffle=False)

        def _f(x, y):
            return x

        return self._model.predict(dataset.map(_f))

    def score(
        self,
        X: Sequence[str] = None,
        y: Union[Sequence[Sequence[str]], Sequence[str]] = None,
        *,
        dataset: NLPDataset = None,
        thresholds: Union[float, Dict[str, float]] = 0.5,
        batch_size: int = 128
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def get_config(self) -> Dict[str, Any]:
        return {
            **super().get_config(),
            "classes": list(self._class2idx.keys()),
            "task": self._task,
            "algorithm": self._algorithm
        }
