from __future__ import annotations
from typing import Optional

import tensorflow as tf
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy

from sknlp.utils.tensor import pad2shape
from .utils import logits2pred


@tf.keras.utils.register_keras_serializable(package="sknlp")
class AccuracyWithLogits(Accuracy):
    def __init__(
        self,
        name: str = "accuracy",
        dtype: Optional[tf.DType] = None,
        activation: str = "linear",
    ) -> None:
        super().__init__(name=name, dtype=dtype)
        self.activation = activation

    def update_state(
        self,
        y_true: tf.Tensor,
        y_logits: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> None:
        y_pred = logits2pred(y_logits, self.activation)
        y_pred = pad2shape(y_pred, tf.shape(y_true), value=0)
        super().update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self) -> dict[str, str]:
        return {**super().get_config(), "activation": self.activation}


@tf.keras.utils.register_keras_serializable(package="sknlp")
class BinaryAccuracyWithLogits(BinaryAccuracy):
    def __init__(
        self,
        name: str = "accuracy",
        dtype: Optional[tf.DType] = None,
        threshold: float = 0.5,
        activation: str = "linear",
    ) -> None:
        super().__init__(name=name, dtype=dtype, threshold=threshold)
        self.activation = activation

    def update_state(
        self,
        y_true: tf.Tensor,
        y_logits: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> None:
        y_pred = logits2pred(y_logits, self.activation)
        y_pred = pad2shape(y_pred, tf.shape(y_true), value=0)
        super().update_state(y_true, y_pred, sample_weight=sample_weight)
