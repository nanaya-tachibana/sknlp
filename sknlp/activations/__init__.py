import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="sknlp")
def gelu(x: tf.Tensor) -> tf.Tensor:
    return tf.keras.activations.gelu(x, approximate=True)