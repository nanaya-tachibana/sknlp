from __future__ import annotations
from typing import Any

import tensorflow as tf

from sknlp.utils.tensor import pad2shape


@tf.keras.utils.register_keras_serializable(package="sknlp")
class MultiLabelCategoricalCrossentropy(tf.keras.losses.Loss):
    """
    苏剑林. (Apr. 25, 2020). 《将“softmax+交叉熵”推广到多标签分类问题 》[Blog post]. Retrieved from https://spaces.ac.cn/archives/7359
    """

    def __init__(
        self,
        reduction: str = tf.keras.losses.Reduction.AUTO,
        flatten_axis: int = 1,
        name: str = "multilabel_categorical_crossentropy",
    ) -> None:
        super().__init__(reduction=reduction, name=name)
        self.flatten_axis = flatten_axis

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if isinstance(y_pred, tf.RaggedTensor):
            y_pred = y_pred.to_tensor(y_pred.dtype.min)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = pad2shape(y_pred, tf.shape(y_true))

        sample_size = tf.reduce_prod(tf.shape(y_true)[: self.flatten_axis])
        y_true = tf.reshape(y_true, (sample_size, -1))
        y_pred = tf.reshape(y_pred, (sample_size, -1))
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg: tf.Tensor = y_pred - y_true * y_pred.dtype.max
        y_pred_pos: tf.Tensor = y_pred - (1 - y_true) * y_pred.dtype.max
        zeros = tf.zeros_like(y_pred[..., :1])  # 用于生成logsum中的1
        y_pred_neg = tf.concat([y_pred_neg, zeros], axis=-1)
        y_pred_pos = tf.concat([y_pred_pos, zeros], axis=-1)
        neg_loss = tf.math.reduce_logsumexp(y_pred_neg, axis=-1)
        pos_loss = tf.math.reduce_logsumexp(y_pred_pos, axis=-1)
        return neg_loss + pos_loss

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "flatten_axis": self.flatten_axis}