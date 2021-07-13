from typing import Callable, Union

import tensorflow as tf


WeightRegularizer = Union[Callable[[tf.Tensor], tf.Tensor], str]
WeightInitializer = Union[Callable[[tf.Tensor, tf.DType], tf.Tensor], str]
WeightConstraint = Union[Callable[[tf.Tensor], tf.Tensor], str]


__all__ = ["WeightRegularizer", "WeightInitializer", "WeightConstraint"]
