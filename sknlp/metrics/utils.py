import tensorflow as tf


def logits2pred(logits: tf.Tensor, activation: str) -> tf.Tensor:
    if isinstance(logits, tf.RaggedTensor):
        fill_value = 0
        if activation in ("sigmoid", "softmax"):
            fill_value = -1e12
        logits = logits.to_tensor(fill_value)
    return tf.keras.activations.get(activation)(logits)