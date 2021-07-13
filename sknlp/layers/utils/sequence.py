from typing import Union, Optional
import tensorflow as tf
import tensorflow.keras.backend as K


def mask_sequence(
    x: tf.Tensor,
    mask: tf.Tensor,
    value: Union[str, float] = 0.0,
    axis: Optional[int] = None,
):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    value: mask部分要被替换成的值，可以是'-inf'或'inf'；
    axis: 序列所在轴，默认为1；
    Copy from https://github.com/bojone/bert4keras/blob/master/bert4keras/backend.py#L117
    """
    if mask is None:
        return x
    else:
        if value == "-inf":
            value = -1e12
        elif value == "inf":
            value = 1e12
        value = K.cast(value, K.dtype(x))
        if K.dtype(mask) != K.dtype(x):
            mask = K.cast(mask, K.dtype(x))
        if axis is None:
            axis = 1
        elif axis < 0:
            axis = K.ndim(x) + axis
        if axis <= 0:
            raise ValueError("axis must be greater than 0")
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        return x * mask + value * (1 - mask)