from typing import Type, Sequence, Dict, Any, List, Optional, Union
import os
import contextlib

import tensorflow as tf
import kerastuner as kt

from sknlp.callbacks import default_supervised_model_callbacks
from sknlp.data import NLPDataset
from .supervised_model import SupervisedNLPModel


@contextlib.contextmanager
def maybe_distribute(distribution_strategy):
    """Distributes if distribution_strategy is set."""
    if distribution_strategy is None:
        yield
    else:
        with distribution_strategy.scope():
            yield


def create_model_builder(
    model_type: Type[SupervisedNLPModel],
    model_args: List[Any],
    model_kwargs: Dict[str, Any],
    optimizer_parameters: Dict[str, Any],
    distribute_strategy: Optional[tf.distribute.Strategy] = None,
):
    def model_builder(hp):
        with maybe_distribute(distribute_strategy):
            model = model_type(*model_args, **model_kwargs)
            model.compile_optimizer(**optimizer_parameters)
            return model._model

    return model_builder


class ParameterSearcher:
    def __init__(
        self,
        model_type: Type[SupervisedNLPModel],
        hyper_parameters: kt.HyperParameters,
        *args,
        **kwargs,
    ) -> None:
        self.model_type = model_type
        self.hyper_parameters = hyper_parameters
        self.model_args = args
        self.model_kwargs = kwargs
        self.temp_model: SupervisedNLPModel = model_type(*args, **kwargs)
        self.tuner = None

    def search(
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
        verbose: int = 2,
        distribute_strategy: Optional[tf.distribute.Strategy] = None,
        max_trials: int = 20,
        executions_per_trial: int = 3,
        search_result_directory: Optional[str] = None,
    ) -> None:
        train_dataset = self.temp_model.prepare_dataset(X, y, dataset)
        assert (
            valid_X is None or valid_y is None
        ) or valid_dataset is None, "No validation set is provided."
        valid_dataset = self.temp_model.prepare_dataset(valid_X, valid_y, valid_dataset)
        train_tf_dataset = train_dataset.batchify(batch_size)
        valid_tf_dataset = valid_dataset.batchify(batch_size, shuffle=False)

        monitor = "val_loss"
        monitor_direction = "min"
        if early_stopping_monitor == 2 and self.temp_model.get_monitor():
            monitor = self.temp_model.get_monitor()
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

        optimizer_parameters = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }
        if clip is not None:
            optimizer_parameters["clipnorm"] = clip
        model_builder = create_model_builder(
            self.model_type,
            self.model_args,
            self.model_kwargs,
            optimizer_parameters,
            distribute_strategy=distribute_strategy,
        )

        monitor = self.temp_model.get_monitor()
        objective = kt.Objective(monitor, monitor_direction)
        directory = None
        project_name = None
        if search_result_directory is not None:
            directory = os.path.dirname(search_result_directory)
            project_name = os.path.basename(search_result_directory)
            if directory == project_name:
                project_name = "default"
        self.tuner = kt.RandomSearch(
            model_builder,
            objective=objective,
            hyperparameters=self.hyper_parameters,
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=directory,
            project_name=project_name,
            distribution_strategy=distribute_strategy,
            overwrite=True,
        )
        self.tuner.search(
            train_tf_dataset,
            epochs=n_epochs,
            validation_data=valid_tf_dataset,
            callbacks=callbacks,
            verbose=verbose,
        )

    def summary(self) -> Dict[str, Any]:
        if self.tuner is not None:
            return self.tuner.results_summary()
        return {}
