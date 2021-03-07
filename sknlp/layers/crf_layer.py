from typing import List, Dict, Any, Optional

import tensorflow as tf
from tensorflow_addons.text import crf_log_likelihood, crf_decode

from sknlp.typing import WeightInitializer


class CrfLossLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_tags: int,
        max_sequence_length: int = 120,
        initializer: WeightInitializer = "Orthogonal",
        name: str = "crf",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_tags = num_tags
        self.max_sequence_length = max_sequence_length
        self.initializer = initializer
        self.transition_weight = self.add_weight(
            shape=(self.num_tags, self.num_tags),
            dtype=tf.float32,
            initializer=tf.keras.initializers.get(initializer),
            name="transision_weight",
        )

    def call(self, inputs: List[tf.Tensor], mask: tf.Tensor) -> tf.Tensor:
        emissions, tag_ids = inputs
        mask = tf.cast(mask, tf.int32)
        sequence_lengths = tf.math.reduce_sum(mask, axis=1)
        likelihoods, _ = crf_log_likelihood(
            emissions, tag_ids, sequence_lengths, self.transition_weight
        )
        loss = tf.math.negative(tf.math.reduce_mean(likelihoods))
        self.add_loss(loss)
        return (
            tf.pad(
                emissions,
                [[0, 0], [0, self.max_sequence_length + 2 - tf.shape(mask)[1]], [0, 0]],
            ),
            tf.pad(
                mask, [[0, 0], [0, self.max_sequence_length + 2 - tf.shape(mask)[1]]]
            ),
        )

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "num_tags": self.num_tags,
            "max_sequence_length": self.max_sequence_length,
            "initializer": self.initializer,
        }


class CrfDecodeLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_tags: int,
        max_sequence_length: int = 120,
        initializer: WeightInitializer = "Orthogonal",
        name: str = "crf",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_tags = num_tags
        self.max_sequence_length = max_sequence_length
        self.initializer = initializer
        self.transition_weight = self.add_weight(
            shape=(self.num_tags, self.num_tags),
            dtype=tf.float32,
            initializer=tf.keras.initializers.get(initializer),
            name="transision_weight",
        )

    def call(self, emissions: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        sequence_lengths = tf.math.reduce_sum(mask, axis=1)
        decoded_tag_ids, _ = crf_decode(
            emissions, self.transition_weight, sequence_lengths
        )
        return tf.pad(
            decoded_tag_ids * mask,
            [[0, 0], [0, self.max_sequence_length + 2 - tf.shape(decoded_tag_ids)[1]]],
            mode="CONSTANT",
        )

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "num_tags": self.num_tags,
            "max_sequence_length": self.max_sequence_length,
            "initializer": self.initializer,
        }
