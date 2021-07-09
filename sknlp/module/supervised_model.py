from __future__ import annotations
from typing import Sequence, Union, Optional, Any

import os
import logging

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd

from sknlp.data import NLPDataset
from sknlp.data.text_segmenter import get_segmenter
from sknlp.vocab import Vocab
from sknlp.callbacks import default_supervised_model_callbacks
from .text2vec import Word2vec, Text2vec
from .base_model import BaseNLPModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
logger.addHandler(stream)


class SupervisedNLPModel(BaseNLPModel):
    def __init__(
        self,
        classes: list[str],
        max_sequence_length: Optional[int] = None,
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
            **kwargs
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
    def classes(self) -> list[str]:
        return list(self._class2idx.keys())

    @property
    def text2vec(self) -> Text2vec:
        return self._text2vec

    @property
    def training_dataset(self) -> NLPDataset:
        return getattr(self, "_training_dataset", None)

    @property
    def validation_dataset(self) -> NLPDataset:
        return getattr(self, "_validation_dataset", None)

    def class2idx(self, class_name: str) -> int:
        return self._class2idx.get(class_name, None)

    def idx2class(self, class_idx: int) -> str:
        return self._idx2class.get(class_idx, None)

    def get_inputs(self) -> tf.Tensor:
        if getattr(self, "inputs", None) is None:
            return self.text2vec.get_inputs()
        else:
            return self.inputs

    def get_outputs(self) -> tf.Tensor:
        return self.build_output_layer(self.build_encode_layer(self.get_inputs()))

    def build(self) -> None:
        if self._built:
            return
        assert self._text2vec is not None
        super().build()

    def create_dataset_from_df(
        self, df: pd.DataFrame, no_label: bool = False
    ) -> NLPDataset:
        raise NotImplementedError()

    def create_dataset_from_csv(
        self, filename: str, no_label: bool = False
    ) -> NLPDataset:
        raise NotImplementedError()

    def prepare_dataset(
        self,
        X: Sequence[str],
        y: Union[Sequence[Sequence[str]], Sequence[str], Sequence[float]],
        dataset: NLPDataset,
    ) -> NLPDataset:
        assert X is not None or dataset is not None
        if dataset is not None:
            return dataset
        if (
            y is not None
            and isinstance(y[0], (list, tuple))
            and isinstance(y[0][0], str)
        ):
            y = ["|".join(map(str, y_i)) for y_i in y]
        if isinstance(X[0], (list, tuple)):
            df = pd.DataFrame(zip(*X, y) if y is not None else X)
        else:
            df = pd.DataFrame(zip(X, y) if y is not None else X)
        return self.create_dataset_from_df(df, y is None)

    def compile_optimizer(self, optimizer_name, **kwargs) -> None:
        if not self._built:
            self.build()
        weight_decay = kwargs.pop("weight_decay", 0)
        if optimizer_name != "adam":
            optimizer = tf.keras.optimizers.deserialize(
                {"class_name": optimizer_name, "config": kwargs}
            )
        elif weight_decay > 0:
            optimizer = tfa.optimizers.AdamW(weight_decay, **kwargs)
        else:
            optimizer = tfa.optimizers.LazyAdam(**kwargs)
        self._model.compile(
            optimizer=optimizer,
            loss=self.get_loss(),
            metrics=self.get_metrics(),
        )

    def fit(
        self,
        X: Sequence[str] = None,
        y: Union[Sequence[Sequence[str]], Sequence[str], Sequence[float]] = None,
        *,
        dataset: NLPDataset = None,
        validation_X: Sequence[str] = None,
        validation_y: Union[
            Sequence[Sequence[str]], Sequence[str], Sequence[float]
        ] = None,
        validation_dataset: NLPDataset = None,
        batch_size: int = 128,
        n_epochs: int = 10,
        optimizer: str = "adam",
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        clip: Optional[float] = 5.0,
        learning_rate_update_factor: float = 0.5,
        learning_rate_update_epochs: int = 10,
        learning_rate_warmup_steps: int = 0,
        enable_early_stopping: bool = False,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        early_stopping_use_best_epoch: bool = False,
        early_stopping_monitor: int = 2,  # 1 for loss, 2 for metric
        checkpoint: Optional[str] = None,
        save_frequency: int = 1,
        log_file: Optional[str] = None,
        verbose: int = 2
    ) -> None:
        if self._text2vec is None:
            cut = get_segmenter(self.segmenter)
            vocab = self.build_vocab(X or dataset.text, cut)
            self._text2vec = Word2vec(vocab, self.embedding_size, self.segmenter)

        self._training_dataset = self.prepare_dataset(X, y, dataset)
        if (
            validation_X is None or validation_y is None
        ) and validation_dataset is None:
            self._validation_dataset = None
        else:
            self._validation_dataset = self.prepare_dataset(
                validation_X, validation_y, validation_dataset
            )
        training_tf_dataset = self.training_dataset.batchify(batch_size)
        validation_tf_dataset = None
        if self.validation_dataset is not None:
            validation_tf_dataset = self.validation_dataset.batchify(
                batch_size, shuffle=False
            )

        optimizer_kwargs = optimizer_kwargs or dict()
        optimizer_kwargs = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            **optimizer_kwargs,
        }
        if clip is not None:
            optimizer_kwargs["clipnorm"] = clip
        self.compile_optimizer(optimizer, **optimizer_kwargs)

        monitor = "val_loss"
        monitor_direction = "min"
        if early_stopping_monitor == 2 and self.get_monitor():
            monitor = self.get_monitor()
            monitor_direction = "max"
        callbacks = default_supervised_model_callbacks(
            learning_rate_update_factor=learning_rate_update_factor,
            learning_rate_update_epochs=learning_rate_update_epochs,
            learning_rate_warmup_steps=learning_rate_warmup_steps,
            use_weight_decay=weight_decay > 0,
            enable_early_stopping=enable_early_stopping,
            early_stopping_monitor=monitor,
            early_stopping_monitor_direction=monitor_direction,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_use_best_epoch=early_stopping_use_best_epoch,
            log_file=log_file,
        )
        for callback in self.get_callbacks():
            callback.validation_data = validation_tf_dataset
            callbacks.append(callback)

        self._model.fit(
            training_tf_dataset,
            epochs=n_epochs,
            validation_data=validation_tf_dataset if self.validation_dataset else None,
            callbacks=callbacks,
            verbose=verbose,
        )

    def predict(
        self,
        X: Sequence[str] = None,
        *,
        dataset: NLPDataset = None,
        thresholds: Union[float, list[float], None] = None,
        batch_size: int = 128
    ) -> Union[np.ndarray, list[str], list[list[str]], list[float]]:
        assert self._built
        dataset = self.prepare_dataset(X, None, dataset)
        return self._model.predict(dataset.batchify(batch_size, shuffle=False))

    def score(
        self,
        X: Sequence[str] = None,
        y: Union[Sequence[Sequence[str]], Sequence[str], Sequence[float]] = None,
        *,
        dataset: NLPDataset = None,
        thresholds: Union[float, list[float], None] = None,
        batch_size: int = 128
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "classes": self.classes,
            "task": self._task,
            "algorithm": self._algorithm,
        }

    def save(self, directory: str) -> None:
        super().save(directory)
        self.text2vec.save_vocab(directory)

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
        self.text2vec.save_vocab(d)

    @classmethod
    def get_custom_objects(cls) -> dict[str, Any]:
        return {**super().get_custom_objects(), "AdamW": tfa.optimizers.AdamW}
