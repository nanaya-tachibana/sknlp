from __future__ import annotations
from typing import Optional, Sequence, Any, Union

import pandas as pd

from sknlp.data import NLPDataset
from sknlp.vocab import Vocab
from .base_model import BaseNLPModel


class UnsupervisedNLPModel(BaseNLPModel):
    def __init__(self, vocab: Vocab, algorithm: Optional[str] = None, **kwargs) -> None:
        super().__init__(vocab, **kwargs)
        self._algorithm = algorithm

    def fit(
        self,
        X: Sequence[Any] = None,
        *,
        dataset: NLPDataset = None,
        validation_X: Optional[Sequence[Any]] = None,
        validation_y: Optional[Sequence[Any]] = None,
        validation_dataset: NLPDataset = None,
        batch_size: int = 128,
        n_epochs: int = 10,
        optimizer: str = "adam",
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0,
        clip: Optional[float] = 1,
        learning_rate_update_factor: float = 0.5,
        learning_rate_update_epochs: int = 10,
        learning_rate_warmup_steps: int = 0,
        enable_early_stopping: bool = False,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0,
        early_stopping_use_best_epoch: bool = False,
        early_stopping_monitor: int = 2,
        checkpoint: Optional[str] = None,
        log_file: Optional[str] = None,
        verbose: int = 2
    ) -> None:
        return super().fit(
            X=X,
            dataset=dataset,
            validation_X=validation_X,
            validation_y=validation_y,
            validation_dataset=validation_dataset,
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clip=clip,
            learning_rate_update_factor=learning_rate_update_factor,
            learning_rate_update_epochs=learning_rate_update_epochs,
            learning_rate_warmup_steps=learning_rate_warmup_steps,
            enable_early_stopping=enable_early_stopping,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_use_best_epoch=early_stopping_use_best_epoch,
            early_stopping_monitor=early_stopping_monitor,
            checkpoint=checkpoint,
            log_file=log_file,
            verbose=verbose,
        )

    def score(
        self,
        X: Sequence[Any] = None,
        *,
        dataset: NLPDataset = None,
        thresholds: Union[float, list[float], None] = None,
        batch_size: int = 128
    ) -> pd.DataFrame:
        return super().score(
            X=X, dataset=dataset, thresholds=thresholds, batch_size=batch_size
        )

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "algorithm": self._algorithm,
        }