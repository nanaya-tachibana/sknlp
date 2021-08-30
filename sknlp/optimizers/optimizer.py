from __future__ import annotations
from typing import Optional, Callable
import functools

import tensorflow as tf
from tensorflow_addons.optimizers import DecoupledWeightDecayExtension


class OptimizerExtension(DecoupledWeightDecayExtension):
    def __init__(
        self,
        *args,
        learning_rate_multiplier: Optional[dict[str, float]] = None,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.learning_rate_multiplier = learning_rate_multiplier
        self._resource_apply_dense = self.learning_rate_multiply(
            self._resource_apply_dense
        )
        self._resource_apply_sparse = self.learning_rate_multiply(
            self._resource_apply_sparse
        )

    def learning_rate_multiply(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(
            *args,
            apply_state: Optional[dict[tuple[str, tf.DType], tf.Tensor]] = None,
            **kwargs
        ) -> tf.Tensor:
            var: tf.Variable = args[1]
            keys = ["lr_t"]
            values = []
            apply_state_is_none = apply_state is None
            if apply_state_is_none:
                apply_state = {
                    (var.device, var.dtype): self._fallback_apply_state(
                        var.device, var.dtype
                    )
                }
            if var.name in self.learning_rate_multiplier:
                if "lr" in apply_state[(var.device, var.dtype)]:
                    keys.append("lr")
                for key in keys:
                    values.append(apply_state[(var.device, var.dtype)][key])
                    apply_state[(var.device, var.dtype)][key] = (
                        self.learning_rate_multiplier[var.name]
                        * apply_state[(var.device, var.dtype)][key]
                    )
            output = func(*args, apply_state=apply_state, **kwargs)
            if not apply_state_is_none and var.name in self.learning_rate_multiplier:
                for key, value in zip(keys, values):
                    apply_state[(var.device, var.dtype)][key] = value
            return output

        return wrapper