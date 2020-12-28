from typing import Sequence, Union, Optional, Dict, Any, List

import os
import logging

import tensorflow as tf
from tensorflow.python.keras.engine import training
import tensorflow_addons as tfa
import pandas as pd

from ..data import NLPDataset
from ..data.text_segmenter import get_segmenter
from ..vocab import Vocab
from ..callbacks import WeightDecayScheduler
from .text2vec import Word2vec, Text2vec
from .base_model import BaseNLPModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
logger.addHandler(stream)


class SupervisedNLPModel(BaseNLPModel):
    def __init__(
        self,
        classes: Sequence[str],
        max_sequence_length: int = None,
        sequence_length: Optional[int] = None,
        segmenter: str = "jieba",
        embedding_size: int = 100,
        text2vec: Optional[Text2vec] = None,
        task: Optional[str] = None,
        algorithm: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(
            max_sequence_length=max_sequence_length,
            sequence_length=sequence_length,
            segmenter=text2vec.segmenter if text2vec else segmenter,
        )
        self._text2vec = text2vec
        if text2vec is not None:
            self._embedding_size = text2vec.embedding_size
            self._max_sequence_length = (
                max_sequence_length or text2vec.max_sequence_length
            )
            self._sequence_length = text2vec.sequence_length
        else:
            self._embedding_size = embedding_size
            self._max_sequence_length = max_sequence_length
            self._sequence_length = sequence_length

        self._task = task
        self._algorithm = algorithm
        self._class2idx = dict(zip(list(classes), range(len(classes))))
        self._idx2class = dict(zip(range(len(classes)), list(classes)))

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def num_classes(self) -> int:
        return len(self._class2idx)

    @property
    def classes(self) -> List[str]:
        return list(self._class2idx.keys())

    def get_inputs(self) -> tf.Tensor:
        return self._text2vec.get_inputs()

    def get_outputs(self) -> tf.Tensor:
        inputs = self._text2vec.get_outputs()
        return self.build_output_layer(self.build_encode_layer(inputs))

    def build(self) -> None:
        if self._built:
            return
        assert self._text2vec is not None
        super().build()

    @classmethod
    def create_dataset_from_df(
        cls, df: pd.DataFrame, vocab: Vocab, segmenter: str, labels: Sequence[str]
    ) -> NLPDataset:
        raise NotImplementedError()

    def prepare_tf_dataset(
        self,
        X: Sequence[str],
        y: Sequence[str],
        dataset: NLPDataset,
        batch_size: int,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        assert not ((X is None or y is None) and dataset is None)

        if self._text2vec is None:
            cut = get_segmenter(self._segmenter)
            vocab = self.build_vocab(X, cut)
            self._text2vec = Word2vec(vocab, self._embedding_size, self._segmenter)

        if dataset is not None:
            return dataset.batchify(batch_size, shuffle=shuffle)

        if isinstance(y[0], (list, tuple)):
            y = ["|".join(map(str, y_i)) for y_i in y]
        df = pd.DataFrame(zip(X, y), columns=["text", "label"])
        dataset = self.create_dataset_from_df(
            df, self._text2vec.vocab, self._text2vec.segmenter, self.classes
        )
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
        weight_decay: float = 1e-4,
        lr_update_factor: float = 0.5,
        lr_update_epochs: int = 10,
        clip: float = 5.0,
        enable_early_stopping: bool = False,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0,
        early_stopping_use_best_epoch: bool = False,
        early_stopping_monitor: int = 2,  # 1 for loss, 2 for metric
        checkpoint: Optional[str] = None,
        save_frequency: int = 1,
        log_file: Optional[str] = None,
        verbose: int = 2
    ) -> None:
        assert not (
            self._text2vec is None and X is None
        ), "When text2vec is not given, X must be provided"

        self.train_dataset = self.prepare_tf_dataset(
            X,
            y,
            dataset,
            batch_size,
        )
        if (valid_X is None or valid_y is None) and valid_dataset is None:
            self.valid_dataset = None
        else:
            self.valid_dataset = self.prepare_tf_dataset(
                valid_X,
                valid_y,
                valid_dataset,
                batch_size,
                shuffle=False,
            )
        if not self._built:
            self.build()
        optimizer = tfa.optimizers.AdamW(weight_decay, learning_rate=lr, clipnorm=clip)
        if optimizer != "adam":
            pass
        loss = self.get_loss()
        metrics = self.get_metrics()
        callbacks = self.get_callbacks()
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        def lr_scheduler(epoch, lr):
            return lr * lr_update_factor ** ((epoch + 1) % lr_update_epochs == 0)

        def wd_scheduler(epoch, wd):
            return wd * lr_update_factor ** ((epoch + 1) % lr_update_epochs == 0)

        lr_decay = tf.keras.callbacks.LearningRateScheduler(
            lr_scheduler, verbose=verbose
        )
        weight_decay = WeightDecayScheduler(wd_scheduler, verbose=verbose)
        callbacks = [*callbacks, lr_decay, weight_decay]
        if enable_early_stopping:
            if early_stopping_monitor == 2 and self.get_monitor():
                mode = "max"
                monitor = self.get_monitor()
            else:
                mode = "min"
                monitor = "val_loss"
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    min_delta=early_stopping_min_delta,
                    patience=early_stopping_patience or lr_update_epochs,
                    mode=mode,
                    restore_best_weights=early_stopping_use_best_epoch,
                )
            )
        if log_file is not None:
            callbacks.append(tf.keras.callbacks.CSVLogger(log_file))
        self._model.fit(
            self.train_dataset,
            epochs=n_epochs,
            validation_data=self.valid_dataset,
            callbacks=callbacks,
            verbose=verbose,
        )

    def dummy_y(self, X: Sequence[str]) -> List[Any]:
        raise NotImplementedError()

    def predict(
        self,
        X: Sequence[str] = None,
        *,
        dataset: NLPDataset = None,
        batch_size: int = 128
    ) -> tf.Tensor:
        assert self._built
        y = None if X is None else self.dummy_y(X)
        dataset = self.prepare_tf_dataset(X, y, dataset, batch_size, shuffle=False)
        return self._model.predict(dataset)

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
            "classes": self.classes,
            "task": self._task,
            "algorithm": self._algorithm,
        }

    def save(self, directory: str) -> None:
        super().save(directory)
        self._text2vec.save_vocab(directory)

    @classmethod
    def load(cls, directory: str) -> "SupervisedNLPModel":
        module = super().load(directory)
        with open(os.path.join(directory, "vocab.json")) as f:
            vocab = Vocab.from_json(f.read())
        module._text2vec = Text2vec(
            vocab,
            segmenter=module.segmenter,
            max_sequence_length=module.max_sequence_length,
            sequence_length=module.sequence_length,
        )
        return module

    def export(self, directory: str, name: str, version: str = "0") -> None:
        super().export(directory, name, version)
        d = os.path.join(directory, name, version)
        self._text2vec.save_vocab(d)

    def get_custom_objects(self) -> Dict[str, Any]:
        return {**super().get_custom_objects(), "AdamW": tfa.optimizers.AdamW}
