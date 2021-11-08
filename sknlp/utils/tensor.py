from __future__ import annotations
from typing import Union

import tensorflow as tf


def pad2shape(
    tensor: tf.Tensor, shape: tf.Tensor, value: Union[tf.Tensor, int, float] = 0
) -> tf.Tensor:
    tensor_shape: tf.Tensor = tf.shape(tensor)
    paddings = shape - tensor_shape
    return tf.cond(
        tf.reduce_any(tf.greater(paddings, 0)),
        lambda: tf.pad(
            tensor,
            tf.concat([tf.zeros_like(paddings)[:, None], paddings[:, None]], 1),
            constant_values=tf.cast(value, tensor.dtype),
        ),
        lambda: tensor,
    )