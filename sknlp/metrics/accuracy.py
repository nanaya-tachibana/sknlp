from typing import Optional, Dict

import functools
import tensorflow as tf
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy


class AccuracyWithLogits(Accuracy):
    def __init__(
        self,
        name: str = "accuracy",
        dtype: Optional[tf.DType] = None,
        logits2scores: str = "sigmoid",
    ) -> None:
        super().__init__(name=name, dtype=dtype)
        self.logits2scores = logits2scores
        if logits2scores == "sigmoid":
            self._l2s = tf.math.sigmoid
        elif logits2scores == "softmax":
            self._l2s = functools.partial(tf.math.softmax, axis=-1)

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> None:
        super().update_state(y_true, self._l2s(y_pred), sample_weight=sample_weight)

    def get_config(self) -> Dict[str, str]:
        return {**super().get_config(), "logits2scores": self.logits2scores}


class BinaryAccuracyWithLogits(BinaryAccuracy):
    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> None:
        super().update_state(
            y_true, tf.math.sigmoid(y_pred), sample_weight=sample_weight
        )
