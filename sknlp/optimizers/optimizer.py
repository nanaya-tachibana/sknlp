from __future__ import annotations
import re
from typing import Optional, Callable, Sequence
import functools

import tensorflow as tf
from tensorflow_addons.optimizers import DecoupledWeightDecayExtension


class OptimizerExtension(DecoupledWeightDecayExtension):
    def __init__(
        self,
        *args,
        exclude_from_weight_decay: Optional[Sequence[str]] = ("bias", "beta", "gamma"),
        learning_rate_multiplier: Optional[dict[str, float]] = None,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if exclude_from_weight_decay is not None:
            self.exclude_from_weight_decay = set(exclude_from_weight_decay)
        else:
            self.exclude_from_weight_decay = set()
        self.learning_rate_multiplier = learning_rate_multiplier
        self._resource_apply_dense = self.learning_rate_multiply(
            self._resource_apply_dense
        )
        self._resource_apply_sparse = self.learning_rate_multiply(
            self._resource_apply_sparse
        )

    def _use_weight_decay(self, variable_name: str) -> bool:
        for pattern in self.exclude_from_weight_decay:
            if re.search(pattern, variable_name) is not None:
                return False
        return True

    def _decay_weights_op(
        self,
        var: tf.Variable,
        apply_state: Optional[dict[tuple[str, tf.DType], tf.Tensor]] = None,
    ):
        if self._use_weight_decay(var.name):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)

            m = self.learning_rate_multiplier.get(var.name, 1)
            return var.assign_sub(
                coefficients["lr_t"] / m * coefficients["wd_t"] * var, self._use_locking
            )
        return tf.no_op()

    def _decay_weights_sparse_op(
        self,
        var: tf.Variable,
        indices: tf.Tensor,
        apply_state: Optional[dict[tuple[str, tf.DType], tf.Tensor]] = None,
    ):
        if self._use_weight_decay(var.name):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)

            m = self.learning_rate_multiplier.get(var.name, 1)
            update = (
                -coefficients["lr_t"]
                / m
                * coefficients["wd_t"]
                * tf.gather(var, indices)
            )
            return self._resource_scatter_add(var, indices, update)
        return tf.no_op()

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