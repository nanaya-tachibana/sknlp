from typing import NewType, Callable, Union

import tensorflow as tf


WeightRegularizer = NewType(
    "WeightRegularizer",
    Union[Callable[[tf.Tensor], tf.Tensor], str]
)

WeightInitializer = NewType(
    "WeightInitializer",
    Union[Callable[[tf.Tensor, tf.DType], tf.Tensor], str]
)

WeightConstraint = NewType(
    "WeightConstraint",
    Union[Callable[[tf.Tensor], tf.Tensor], str]
)


__all__ = ["WeightRegularizer", "WeightInitializer", "WeightConstraint"]
