from __future__ import annotations
from typing import Sequence, Optional, Any, Callable

import json
import os
import itertools
from collections import Counter

import numpy as np
import pandas as pd
from tabulate import tabulate

import tensorflow as tf

import sknlp
from sknlp.callbacks import default_supervised_model_callbacks
from sknlp.optimizers import AdamOptimizer
from sknlp.vocab import Vocab
from sknlp.data import NLPDataset


class BaseNLPModel:
    dataset_class = NLPDataset
    dataset_args = ["text_normalization"]

    def __init__(
        self,
        vocab: Vocab,
        max_sequence_length: Optional[int] = None,
        sequence_length: Optional[int] = None,
        segmenter: Optional[str] = None,
        text_normalization: dict[str, str] = {"letter_case": "sensitive"},
        learning_rate_multiplier: Optional[dict[str, float]] = None,
        inference_kwargs: Optional[dict[str, Any]] = None,
        custom_kwargs: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs
    ) -> None:
        self._vocab = vocab
        self._max_sequence_length = max_sequence_length
        self._sequence_length = sequence_length
        self._segmenter = segmenter
        self._text_normalization = text_normalization
        self._learning_rate_multiplier = learning_rate_multiplier or dict()
        self._layerwise_learning_rate_multiplier: list[
            tuple[tf.keras.layers.Layer, float]
        ] = []
        self._inference_kwargs = inference_kwargs or dict()
        self._custom_kwargs = custom_kwargs or dict()
        self._kwargs = kwargs

        self._name = name
        if getattr(self, "evaluation_dataset_class", None) is None:
            self.evaluation_dataset_class = self.dataset_class

        self._model: Optional[tf.keras.Model] = None
        self._inference_model: Optional[tf.keras.Model] = None
        self._built = False

    @property
    def dataset_kwargs(self) -> dict[str, Any]:
        kwargs = dict()
        for field in self.dataset_args:
            instance_field = getattr(self, field, None)
            if instance_field is not None:
                kwargs[field] = instance_field
        return kwargs

    @property
    def name(self) -> str:
        return self._name

    @property
    def vocab(self) -> Vocab:
        return self._vocab

    @property
    def max_sequence_length(self) -> Optional[int]:
        return self._max_sequence_length

    @property
    def sequence_length(self) -> Optional[int]:
        return self._sequence_length

    @property
    def segmenter(self) -> Optional[str]:
        return self._segmenter

    @property
    def text_normalization(self) -> dict[str, str]:
        return self._text_normalization

    @property
    def learning_rate_multiplier(self) -> dict[str, float]:
        for layer, multiplier in self._layerwise_learning_rate_multiplier:
            for variable in layer.variables:
                self._learning_rate_multiplier[variable.name] = multiplier
        return self._learning_rate_multiplier

    @staticmethod
    def build_vocab(
        texts: Sequence[str],
        segment_func: Callable[[str], Sequence[str]],
        min_frequency=5,
    ) -> Vocab:
        counter = Counter(
            itertools.chain.from_iterable(segment_func(text) for text in texts)
        )
        return Vocab(counter, min_frequency=min_frequency)

    def build(self) -> None:
        if self._built:
            return
        self._model = tf.keras.Model(
            inputs=self.get_inputs(), outputs=self.get_outputs(), name=self.name
        )
        self._built = True
        self.build_inference_model()

    def build_inference_model(self) -> tf.keras.Model:
        self._inference_model = self._model

    def build_encoding_layer(
        self, inputs: tf.Tensor | list[tf.Tensor]
    ) -> tf.Tensor | list[tf.Tensor]:
        return inputs

    def build_intermediate_layer(
        self, inputs: tf.Tensor | list[tf.Tensor]
    ) -> tf.Tensor | list[tf.Tensor]:
        return inputs

    def build_output_layer(
        self, inputs: tf.Tensor | list[tf.Tensor]
    ) -> tf.Tensor | list[tf.Tensor]:
        raise NotImplementedError()

    def get_inputs(self) -> list[tf.Tensor] | tf.Tensor:
        raise NotImplementedError()

    def get_outputs(self) -> list[tf.Tensor] | tf.Tensor:
        return self.build_output_layer(
            self.build_intermediate_layer(self.build_encoding_layer(self.get_inputs()))
        )

    def get_loss(self, *args, **kwargs) -> Optional[tf.keras.losses.Loss]:
        raise NotImplementedError()

    def get_metrics(self, *args, **kwargs) -> list[tf.keras.metrics.Metric]:
        return []

    def get_callbacks(self, *args, **kwargs) -> list[tf.keras.callbacks.Callback]:
        return []

    def get_monitor(self) -> str:
        raise NotImplementedError()

    @classmethod
    def _get_model_filename_template(cls) -> str:
        return "model_{epoch:04d}"

    @classmethod
    def _get_model_filename(cls, epoch: Optional[int] = None) -> str:
        if epoch is not None:
            if epoch < 1:
                epoch = 0
            return cls._get_model_filename_template().format(epoch=epoch)
        return "model"

    def freeze(self) -> None:
        for layer in self._model.layers:
            layer.trainable = False

    def create_dataset_from_csv(
        self, filename: str, has_label: bool = True, evaluation: bool = False
    ) -> NLPDataset:
        dataset_class = self.dataset_class
        if evaluation:
            dataset_class = self.evaluation_dataset_class
        return dataset_class(
            self.vocab,
            self.classes,
            segmenter=self.segmenter,
            csv_file=filename,
            max_length=self.max_sequence_length,
            has_label=has_label,
            **self.dataset_kwargs
        )

    def prepare_dataset(
        self,
        X: Sequence[Any],
        y: Sequence[Any],
        dataset: NLPDataset,
        evaluation: bool = False,
    ) -> NLPDataset:
        assert X is not None or dataset is not None
        if dataset is not None:
            return dataset
        dataset_class = self.dataset_class
        if evaluation:
            dataset_class = self.evaluation_dataset_class
        return dataset_class(
            self.vocab,
            self.classes,
            segmenter=self.segmenter,
            X=X,
            y=y,
            max_length=self.max_sequence_length,
            has_label=y is not None,
            **self.dataset_kwargs
        )

    def compile_optimizer(self, optimizer_name, **kwargs) -> None:
        if not self._built:
            self.build()

        weight_decay = kwargs.pop("weight_decay", 0)
        if optimizer_name != "adam":
            optimizer = tf.keras.optimizers.deserialize(
                {"class_name": optimizer_name, "config": kwargs}
            )
        else:
            optimizer = AdamOptimizer(
                weight_decay,
                learning_rate_multiplier=self.learning_rate_multiplier,
                **kwargs
            )
        self._model.compile(
            optimizer=optimizer,
            loss=self.get_loss(),
            metrics=self.get_metrics(),
        )

    def fit(
        self,
        X: Sequence[Any] = None,
        y: Sequence[Any] = None,
        *,
        dataset: NLPDataset = None,
        validation_X: Optional[Sequence[Any]] = None,
        validation_y: Optional[Sequence[Any]] = None,
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
        early_stopping_use_best_epoch: bool = True,
        early_stopping_monitor: int = 2,  # 1 for loss, 2 for metric
        checkpoint: Optional[str] = None,
        log_file: Optional[str] = None,
        verbose: int = 2
    ) -> None:
        training_dataset = self.prepare_dataset(X, y, dataset)
        if (
            validation_X is None or validation_y is None
        ) and validation_dataset is None:
            validation_dataset = None
        else:
            validation_dataset = self.prepare_dataset(
                validation_X, validation_y, validation_dataset
            )
        training_tf_dataset = training_dataset.batchify(batch_size)
        validation_tf_dataset = None
        if validation_dataset is not None:
            validation_tf_dataset = validation_dataset.batchify(
                batch_size, shuffle=False
            )

        optimizer_kwargs = optimizer_kwargs or dict()
        optimizer_kwargs = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            **optimizer_kwargs,
        }
        if clip is not None:
            optimizer_kwargs["global_clipnorm"] = clip
        self.compile_optimizer(optimizer, **optimizer_kwargs)

        monitor = "val_loss"
        monitor_direction = "min"
        if early_stopping_monitor == 2 and self.get_monitor():
            monitor = self.get_monitor()
            monitor_direction = "max"

        has_validation_dataset = validation_tf_dataset is not None
        if checkpoint is not None:
            if has_validation_dataset:
                checkpoint = os.path.join(checkpoint, self._get_model_filename(epoch=0))
            else:
                checkpoint = os.path.join(
                    checkpoint, self._get_model_filename_template()
                )
        callbacks = default_supervised_model_callbacks(
            learning_rate_update_factor=learning_rate_update_factor,
            learning_rate_update_epochs=learning_rate_update_epochs,
            learning_rate_warmup_steps=learning_rate_warmup_steps,
            has_validation_dataset=has_validation_dataset,
            enable_early_stopping=enable_early_stopping,
            early_stopping_monitor=monitor,
            early_stopping_monitor_direction=monitor_direction,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_use_best_epoch=early_stopping_use_best_epoch,
            checkpoint=checkpoint,
            log_file=log_file,
            verbose=verbose,
        )
        for callback in self.get_callbacks():
            callback.validation_data = validation_tf_dataset
            callbacks.append(callback)

        self._model.fit(
            x=training_tf_dataset,
            y=None,
            epochs=n_epochs,
            validation_data=validation_tf_dataset if has_validation_dataset else None,
            callbacks=callbacks,
            verbose=verbose,
        )
        self.build_inference_model()

    def predict(
        self,
        X: Sequence[Any] = None,
        *,
        dataset: NLPDataset = None,
        thresholds: float | list[float] | None = None,
        batch_size: int = 128
    ) -> np.ndarray | Sequence[Any]:
        assert self._built
        dataset = self.prepare_dataset(X, None, dataset, evaluation=True)
        return self._inference_model.predict(
            dataset.batchify(batch_size, shuffle=False, training=False)
        )

    def score(
        self,
        X: Sequence[Any] = None,
        y: Sequence[Any] = None,
        *,
        dataset: NLPDataset = None,
        thresholds: float | list[float] | None = None,
        batch_size: int = 128
    ) -> pd.DataFrame:
        raise NotImplementedError()

    @classmethod
    def format_score(self, score_df: pd.DataFrame, format: str = "markdown") -> str:
        return tabulate(score_df, headers="keys", tablefmt="github", showindex=False)

    def save_config(self, directory: str, filename: str = "meta.json") -> None:
        with open(os.path.join(directory, filename), "w", encoding="UTF-8") as f:
            f.write(json.dumps(self.get_config(), ensure_ascii=False))

    def save_vocab(self, directory: str, filename: str = "vocab.json") -> None:
        with open(os.path.join(directory, filename), "w", encoding="UTF-8") as f:
            f.write(self.vocab.to_json())

    def save(self, directory: str) -> None:
        options = tf.saved_model.SaveOptions(experimental_custom_gradients=False)
        self._model.save(
            os.path.join(directory, self._get_model_filename()),
            save_format="tf",
            options=options,
        )
        self.save_config(directory)
        self.save_vocab(directory)

    @classmethod
    def load(cls, directory: str, epoch: Optional[int] = None) -> "BaseNLPModel":
        with open(os.path.join(directory, "meta.json"), encoding="UTF-8") as f:
            meta = json.loads(f.read())
        with open(os.path.join(directory, "vocab.json")) as f:
            vocab = Vocab.from_json(f.read())
        module = cls.from_config({"vocab": vocab, **meta})
        module._model = tf.keras.models.load_model(
            os.path.join(directory, cls._get_model_filename(epoch=epoch))
        )
        module._built = True
        return module

    def export(self, directory: str, name: str, version: str = "0") -> None:
        d = os.path.join(directory, name, version)

        model: tf.keras.Model = tf.keras.models.model_from_json(
            self._inference_model.to_json(),
            custom_objects={
                "TruncatedNormal": tf.keras.initializers.TruncatedNormal,
                "GlorotUniform": tf.keras.initializers.GlorotUniform,
                "Orthogonal": tf.keras.initializers.Orthogonal,
                "Zeros": tf.keras.initializers.Zeros,
            },
        )
        self._inference_kwargs["input_names"] = model.input_names
        self._inference_kwargs["output_names"] = model.output_names
        model.set_weights(self._inference_model.get_weights())
        options = tf.saved_model.SaveOptions(
            namespace_whitelist=["Addons"], experimental_custom_gradients=False
        )
        model.save(d, include_optimizer=False, save_format="tf", options=options)
        self.save_config(d)
        self.save_vocab(d)

    def get_config(self) -> dict[str, Any]:
        return {
            "max_sequence_length": self.max_sequence_length,
            "sequence_length": self.sequence_length,
            "segmenter": self.segmenter,
            "name": self.name,
            "text_normalization": self._text_normalization,
            "inference_kwargs": self._inference_kwargs,
            "custom_kwargs": self._custom_kwargs,
            "__version__": sknlp.__version__,
            "__package__": sknlp.__name__,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseNLPModel":
        return cls(**config)