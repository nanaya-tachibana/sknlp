from typing import List, Optional, Callable

import tensorflow as tf

from .model_score import ModelScoreCallback
from .save import ModelSave
from .weight_decay import WeightDecayScheduler


def create_exponential_decay_scheduler(
    update_factor: float, update_epochs: int
) -> Callable[[int, float], float]:
    def scheduler(epoch, x):
        return x * update_factor ** ((epoch + 1) % update_epochs == 0)

    return scheduler


def default_supervised_model_callbacks(
    learning_rate_update_factor: float,
    learning_rate_update_epochs: int,
    use_weight_decay: bool = False,
    enable_early_stopping: bool = False,
    early_stopping_monitor: str = "val_loss",
    early_stopping_monitor_direction: str = "min",
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.0,
    early_stopping_use_best_epoch: bool = False,
    log_file: str = None,
    verbose: int = 2,
) -> List[tf.keras.callbacks.Callback]:
    scheduler = create_exponential_decay_scheduler(
        learning_rate_update_factor, learning_rate_update_epochs
    )
    learning_rate_decay = tf.keras.callbacks.LearningRateScheduler(
        scheduler, verbose=verbose
    )
    callbacks = [learning_rate_decay]
    if use_weight_decay > 0:
        callbacks.append(WeightDecayScheduler(scheduler, verbose=verbose))

    if enable_early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_monitor,
                min_delta=early_stopping_min_delta,
                patience=early_stopping_patience or learning_rate_update_epochs,
                mode=early_stopping_monitor_direction,
                restore_best_weights=early_stopping_use_best_epoch,
            )
        )
    if log_file is not None:
        callbacks.append(tf.keras.callbacks.CSVLogger(log_file))
    return callbacks


__all__ = [
    "ModelScoreCallback",
    "WeightDecayScheduler",
    "ModelSave",
    "default_supervised_model_callbacks",
]
