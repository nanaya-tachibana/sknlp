from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy


@tf.keras.utils.register_keras_serializable(package="sknlp")
class BertLMLossLayer(tf.keras.layers.Layer):
    def __init__(self, name: str = "bert_lm_loss", **kwargs) -> None:
        super().__init__(name=name, **kwargs)

    def call(self, inputs: list[tf.Tensor], *args, **kwargs) -> tf.Tensor:
        # (batch_size, label_seq_len)
        # (batch_size, label_seq_len, vocab_size)
        # (batch_size, label_seq_len)
        token_ids, logits, mask = inputs

        loss = sparse_categorical_crossentropy(token_ids, logits, from_logits=True)
        mask = tf.cast(mask, loss.dtype)
        self.add_loss(tf.reduce_sum(loss * mask) / tf.reduce_sum(mask))
        predicted_token_ids = tf.math.argmax(logits, axis=-1)
        is_equal = tf.cast(tf.equal(token_ids, predicted_token_ids), loss.dtype)
        lm_accuracy = tf.reduce_sum(is_equal * mask) / tf.reduce_sum(mask)
        self.add_metric(lm_accuracy, name="lm_accuracy")
        boolean_mask = tf.cast(mask, tf.bool)
        return tf.ragged.boolean_mask(predicted_token_ids, boolean_mask)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape([input_shape[0][0], None])