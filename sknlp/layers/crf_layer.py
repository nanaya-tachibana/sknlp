from __future__ import annotations
from typing import Any

import tensorflow as tf
from tensorflow.python.ops.array_ops import boolean_mask
from tensorflow_addons.text import crf_log_likelihood, crf_decode

from sknlp.typing import WeightInitializer


@tf.keras.utils.register_keras_serializable(package="sknlp")
class CrfLossLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_tags: int,
        initializer: WeightInitializer = "Orthogonal",
        name: str = "crf",
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.num_tags = num_tags
        self.initializer = initializer

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        self.transition_weight = self.add_weight(
            shape=(self.num_tags, self.num_tags),
            dtype=tf.float32,
            initializer=tf.keras.initializers.get(self.initializer),
            name="transision_weight",
        )

    def call(self, inputs: list[tf.Tensor], mask: tf.Tensor) -> tf.RaggedTensor:
        emissions, tag_ids = inputs
        mask = tf.cast(mask, tf.int32)
        sequence_lengths = tf.math.reduce_sum(mask, axis=1)
        likelihoods, _ = crf_log_likelihood(
            emissions,
            tag_ids,
            sequence_lengths,
            self.transition_weight,
        )
        loss = tf.math.negative(tf.math.reduce_mean(likelihoods))
        self.add_loss(loss)

        decoded_tag_ids, _ = crf_decode(
            emissions,
            self.transition_weight,
            sequence_lengths,
        )
        is_equal = tf.cast(tf.equal(tag_ids, decoded_tag_ids), tf.int32)
        tag_accuracy = tf.reduce_sum(is_equal * mask) / tf.reduce_sum(mask)
        self.add_metric(tag_accuracy, name="tag_accuracy")
        boolean_mask = tf.cast(mask, tf.bool)
        return tf.ragged.boolean_mask(decoded_tag_ids, boolean_mask)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "num_tags": self.num_tags,
            "initializer": self.initializer,
        }


@tf.keras.utils.register_keras_serializable(package="sknlp")
class CrfDecodeLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_tags: int,
        initializer: WeightInitializer = "Orthogonal",
        name: str = "crf",
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.num_tags = num_tags
        self.initializer = initializer

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        self.transition_weight = self.add_weight(
            shape=(self.num_tags, self.num_tags),
            dtype=tf.float32,
            initializer=tf.keras.initializers.get(self.initializer),
            name="transision_weight",
        )

    def call(self, emissions: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        sequence_lengths = tf.math.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        decoded_tag_ids, _ = crf_decode(
            emissions,
            self.transition_weight,
            sequence_lengths,
        )
        boolean_mask = tf.cast(mask, tf.bool)
        return tf.ragged.boolean_mask(decoded_tag_ids, boolean_mask)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "num_tags": self.num_tags,
            "initializer": self.initializer,
        }