from __future__ import annotations
from typing import Optional, Callable

import tensorflow as tf

from .learning_rate_scheduler import LearningRateScheduler

from .tagging_fscore_metric import TaggingFScoreMetric


def create_exponential_decay_scheduler(
    update_factor: float,
    update_epochs: int,
) -> Callable[[int, float], float]:
    def scheduler(epoch: int, x: float) -> float:
        return x * update_factor ** (epoch % update_epochs == 0 and epoch != 0)

    return scheduler


def create_warmup_scheduler(warmup_steps: int) -> Callable[[int, float], float]:
    def scheduler(step: int, x: float) -> float:
        if step == 0 and warmup_steps > 0:
            return x / warmup_steps
        if step >= warmup_steps:
            return x
        return x * (step + 1) / step

    return scheduler


def default_supervised_model_callbacks(
    learning_rate_update_factor: float,
    learning_rate_update_epochs: int,
    learning_rate_warmup_steps: int,
    has_validation_dataset: bool = False,
    enable_early_stopping: bool = False,
    early_stopping_monitor: str = "val_loss",
    early_stopping_monitor_direction: str = "min",
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.0,
    early_stopping_use_best_epoch: bool = False,
    checkpoint: Optional[str] = None,
    log_file: str = None,
    verbose: int = 2,
) -> list[tf.keras.callbacks.Callback]:
    decay_scheduler = create_exponential_decay_scheduler(
        learning_rate_update_factor,
        learning_rate_update_epochs,
    )
    warmup_scheduler = create_warmup_scheduler(learning_rate_warmup_steps)
    learning_rate_decay = LearningRateScheduler(
        warmup_scheduler, decay_scheduler, verbose=verbose
    )
    callbacks = [learning_rate_decay]

    if enable_early_stopping and has_validation_dataset:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_monitor,
                min_delta=early_stopping_min_delta,
                patience=early_stopping_patience or learning_rate_update_epochs,
                mode=early_stopping_monitor_direction,
                restore_best_weights=early_stopping_use_best_epoch,
                verbose=verbose,
            )
        )
    if checkpoint is not None:
        options = tf.saved_model.SaveOptions(experimental_custom_gradients=False)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint,
                monitor=early_stopping_monitor,
                mode=early_stopping_monitor_direction,
                save_best_only=has_validation_dataset,
                options=options,
                verbose=verbose,
            )
        )
    if log_file is not None:
        callbacks.append(tf.keras.callbacks.CSVLogger(log_file))
    return callbacks


__all__ = [
    "LearningRateSchduler",
    "TaggingFScoreMetric",
    "default_supervised_model_callbacks",
]