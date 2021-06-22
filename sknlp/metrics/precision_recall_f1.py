from __future__ import annotations
from typing import Optional, Any

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from tensorflow_addons.metrics import FBetaScore


class PrecisionWithLogits(Precision):
    def __init__(
        self,
        top_k: Optional[int] = None,
        class_id: Optional[int] = None,
        threshold: float = 0.5,
        name: str = "precision",
        dtype: Optional[tf.DType] = None,
        activation: str = "linear",
    ) -> None:
        super().__init__(
            thresholds=threshold,
            top_k=top_k,
            class_id=class_id,
            name=name,
            dtype=dtype,
        )
        self.activation = activation

    def update_state(
        self,
        y_true: tf.Tensor,
        y_logits: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> None:
        y_pred = tf.keras.activations.deserialize(self.activation)(y_logits)
        super().update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self) -> dict[str, Any]:
        configs = super().get_config()
        configs.pop("thresholds", None)
        return {
            **configs,
            "threshold": self.thresholds,
            "activation": self.activation,
        }


class RecallWithLogits(Recall):
    def __init__(
        self,
        top_k: Optional[int] = None,
        class_id: Optional[int] = None,
        threshold: float = 0.5,
        name: str = "recall",
        dtype: Optional[tf.DType] = None,
        activation: str = "linear",
    ) -> None:
        super().__init__(
            thresholds=threshold,
            top_k=top_k,
            class_id=class_id,
            name=name,
            dtype=dtype,
        )
        self.activation = activation

    def update_state(
        self,
        y_true: tf.Tensor,
        y_logits: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> None:
        y_pred = tf.keras.activations.deserialize(self.activation)(y_logits)
        super().update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self) -> dict[str, Any]:
        configs = super().get_config()
        configs.pop("thresholds", None)
        return {
            **configs,
            "threshold": self.thresholds,
            "activation": self.activation,
        }


class FBetaScoreWithLogits(FBetaScore):
    def __init__(
        self,
        num_classes: int,
        average: str = "micro",
        beta: float = 1.0,
        threshold: float = 0.5,
        name: str = "fbeta_score",
        dtype: Optional[tf.DType] = None,
        activation: str = "linear",
    ) -> None:
        super().__init__(
            num_classes,
            average=average,
            beta=beta,
            threshold=threshold,
            name=name,
            dtype=dtype,
        )
        self.activation = activation

    def update_state(
        self,
        y_true: tf.Tensor,
        y_logits: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> None:
        y_pred = tf.keras.activations.deserialize(self.activation)(y_logits)
        super().update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "activation": self.activation,
        }