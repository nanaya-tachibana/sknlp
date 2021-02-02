from typing import Sequence, Union, Optional, Dict, Any, List, Tuple, Type

import os
import logging

import tensorflow as tf
import tensorflow_addons as tfa
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

    @property
    def text2vec(self) -> Text2vec:
        return self._text2vec

    @property
    def train_dataset(self) -> tf.data.Dataset:
        return getattr(self, "_train_dataset", None)

    @property
    def valid_dataset(self) -> tf.data.Dataset:
        return getattr(self, "_valid_dataset", None)

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
        self, df: pd.DataFrame, vocab: Vocab, segmenter: str, labels: Sequence[str]
    ) -> NLPDataset:
        raise NotImplementedError()

    def prepare_dataset(
        self,
        X: Sequence[str],
        y: Sequence[str],
        dataset: NLPDataset,
    ) -> NLPDataset:
        assert (X is not None and y is not None) or dataset is not None
        if dataset is not None:
            return dataset
        if isinstance(y[0], (list, tuple)):
            y = ["|".join(map(str, y_i)) for y_i in y]
        df = pd.DataFrame(zip(X, y), columns=["text", "label"])
        return self.create_dataset_from_df(
            df, self.text2vec.vocab, self.text2vec.segmenter, self.classes
        )

    def compile_optimizer(self, **kwargs) -> None:
        if not self._built:
            self.build()
        weight_decay = kwargs.pop("weight_decay", 0)
        if weight_decay > 0:
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
        y: Union[Sequence[Sequence[str]], Sequence[str]] = None,
        *,
        dataset: NLPDataset = None,
        valid_X: Sequence[str] = None,
        valid_y: Union[Sequence[Sequence[str]], Sequence[str]] = None,
        valid_dataset: NLPDataset = None,
        batch_size: int = 128,
        n_epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        clip: Optional[float] = 5.0,
        learning_rate_update_factor: float = 0.5,
        learning_rate_update_epochs: int = 10,
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

        self._train_dataset = self.prepare_dataset(X, y, dataset)
        if (valid_X is None or valid_y is None) and valid_dataset is None:
            self._valid_dataset = None
        else:
            self._valid_dataset = self.prepare_dataset(valid_X, valid_y, valid_dataset)
        train_tf_dataset = self.train_dataset.batchify(batch_size)
        valid_tf_dataset = None
        if self.valid_dataset is not None:
            valid_tf_dataset = self.valid_dataset.batchify(batch_size, shuffle=False)

        optimizer_kwargs = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }
        if clip is not None:
            optimizer_kwargs["clipnorm"] = clip
        self.compile_optimizer(**optimizer_kwargs)

        monitor = "val_loss"
        monitor_direction = "min"
        if early_stopping_monitor == 2 and self.get_monitor():
            monitor = self.get_monitor()
            monitor_direction = "max"
        callbacks = default_supervised_model_callbacks(
            learning_rate_update_factor=learning_rate_update_factor,
            learning_rate_update_epochs=learning_rate_update_epochs,
            use_weight_decay=weight_decay > 0,
            enable_early_stopping=enable_early_stopping,
            early_stopping_monitor=monitor,
            early_stopping_monitor_direction=monitor_direction,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_use_best_epoch=early_stopping_use_best_epoch,
            log_file=log_file,
        )
        callbacks.extend(self.get_callbacks(batch_size))

        self._model.fit(
            train_tf_dataset,
            epochs=n_epochs,
            validation_data=valid_tf_dataset if self.valid_dataset else None,
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
        dataset = self.prepare_dataset(X, y, dataset)
        return self._model.predict(dataset.batchify(batch_size, shuffle=False))

    def score(
        self,
        X: Sequence[str] = None,
        y: Union[Sequence[Sequence[str]], Sequence[str]] = None,
        *,
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
    def get_custom_objects(cls) -> Dict[str, Any]:
        return {**super().get_custom_objects(), "AdamW": tfa.optimizers.AdamW}
