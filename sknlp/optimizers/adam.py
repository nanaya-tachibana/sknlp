from __future__ import annotations
from typing import Optional, Sequence

import tensorflow as tf

from .optimizer import OptimizerExtension


@tf.keras.utils.register_keras_serializable(package="sknlp")
class AdamOptimizer(OptimizerExtension, tf.keras.optimizers.Adam):
    def __init__(
        self,
        weight_decay: float,
        exclude_from_weight_decay: Optional[Sequence[str]] = ("bias", "beta", "gamma"),
        learning_rate_multiplier: Optional[dict[str, float]] = None,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-06,
        amsgrad: bool = False,
        name: str = "Adam",
        **kwargs,
    ) -> None:
        super().__init__(
            weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            learning_rate_multiplier=learning_rate_multiplier,
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            **kwargs,
        )
