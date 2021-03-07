from typing import Sequence, Optional, Union, Any, Dict
import functools

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from tensorflow_addons.metrics import FBetaScore


class PrecisionWithLogits(Precision):
    def __init__(
        self,
        thresholds: Union[float, Sequence[float], None] = None,
        top_k: Optional[int] = None,
        class_id: Optional[int] = None,
        name: str = "precision",
        dtype: Optional[tf.DType] = None,
        logits2scores: str = "sigmoid",
    ) -> None:
        super().__init__(
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            name=name,
            dtype=dtype,
        )
        self.logits2scores = logits2scores
        if logits2scores == "sigmoid":
            self._l2s = tf.math.sigmoid
        elif logits2scores == "softmax":
            self._l2s = functools.partial(tf.math.softmax, axis=-1)

    def update_state(
        self,
        y_true: tf.Tensor,
        y_logits: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> None:
        super().update_state(y_true, self._l2s(y_logits), sample_weight=sample_weight)

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "logits2scores": self.logits2scores}


class RecallWithLogits(Recall):
    def __init__(
        self,
        thresholds: Union[float, Sequence[float], None] = None,
        top_k: Optional[int] = None,
        class_id: Optional[int] = None,
        name: str = "recall",
        dtype: Optional[tf.DType] = None,
        logits2scores: str = "sigmoid",
    ) -> None:
        super().__init__(
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            name=name,
            dtype=dtype,
        )
        self.logits2scores = logits2scores
        if logits2scores == "sigmoid":
            self._l2s = tf.math.sigmoid
        elif logits2scores == "softmax":
            self._l2s = functools.partial(tf.math.softmax, axis=-1)

    def update_state(
        self,
        y_true: tf.Tensor,
        y_logits: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> None:
        super().update_state(y_true, self._l2s(y_logits), sample_weight=sample_weight)

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "logits2scores": self.logits2scores}


class FBetaScoreWithLogits(FBetaScore):
    def __init__(
        self,
        num_classes: int,
        average: str = "micro",
        beta: float = 1.0,
        thresholds: float = 0.5,
        name: str = "fbeta_score",
        dtype: Optional[tf.DType] = None,
        logits2scores: str = "sigmoid",
    ) -> None:
        super().__init__(
            num_classes,
            average=average,
            beta=beta,
            threshold=thresholds,
            name=name,
            dtype=dtype,
        )
        self.logits2scores = logits2scores
        if logits2scores == "sigmoid":
            self._l2s = tf.math.sigmoid
        elif logits2scores == "softmax":
            self._l2s = functools.partial(tf.math.softmax, axis=-1)

    def update_state(
        self,
        y_true: tf.Tensor,
        y_logits: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> None:
        super().update_state(y_true, self._l2s(y_logits), sample_weight=sample_weight)

    def get_config(self) -> Dict[str, Any]:
        configs = super().get_config()
        configs.pop("threshold", None)
        return {
            **configs,
            "logits2scores": self.logits2scores,
            "thresholds": self.threshold,
        }