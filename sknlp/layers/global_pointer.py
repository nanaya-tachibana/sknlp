from __future__ import annotations
from typing import Optional, Any
import tensorflow as tf

from .embedding_layer import SinusoidalPositionEmbedding


class GlobalPointerLayer(tf.keras.layers.Layer):
    """
    苏剑林. (May. 01, 2021). 《GlobalPointer：用统一的方式处理嵌套和非嵌套NER 》[Blog post]. Retrieved from https://spaces.ac.cn/archives/8373
    """

    def __init__(
        self,
        heads: int,
        head_size: int,
        max_sequence_length: int,
        name: str = "global_pointer",
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.heads = heads
        self.head_size = head_size
        self.max_sequence_length = max_sequence_length

    def build(self, input_shape: tf.TensorShape):
        self.dense = tf.keras.layers.Dense(self.head_size * self.heads * 2)
        self.sinusoial_postion = SinusoidalPositionEmbedding(
            self.head_size, self.max_sequence_length
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        实现参照:https://github.com/bojone/bert4keras/blob/b3c28c3a7219c5a152082b1159af0e828923364d/bert4keras/layers.py#L1158
        """
        # (batch, seq, dim) -> (batch, seq, heads * head_size * 2)
        inputs = self.dense(inputs)
        # heads * (batch, seq, head_size * 2)
        inputs = tf.split(inputs, self.heads, axis=-1)
        # (batch, seq, heads, head_size * 2)
        inputs = tf.stack(inputs, axis=-2)
        # (batch, seq, heads, head_size)
        qw, kw = inputs[..., : self.head_size], inputs[..., self.head_size :]

        input_shape = tf.shape(inputs)
        # (1, seq)
        position_ids = tf.range(0, input_shape[1], dtype=tf.int64)[None]
        # (batch, seq)
        position_ids = tf.repeat(position_ids, input_shape[0], axis=0)
        # (bacth, seq, head_size)
        position_embedding = self.sinusoial_postion(position_ids)
        # (batch, seq, 1, head_size)
        sin_position = tf.repeat(position_embedding[..., None, ::2], 2, -1)
        cos_position = tf.repeat(position_embedding[..., None, 1::2], 2, -1)
        # (batch, seq, heads, head_size / 2, 2)
        qw2 = tf.stack([-qw[..., 1::2], qw[..., ::2]], axis=4)
        qw2 = tf.reshape(qw2, tf.shape(qw))
        qw = qw * cos_position + qw2 * sin_position
        kw2 = tf.stack([-kw[..., 1::2], kw[..., ::2]], axis=4)
        kw2 = tf.reshape(kw2, tf.shape(kw))
        kw = kw * cos_position + kw2 * sin_position

        logits = tf.einsum("bmhs,bnhs->bhmn", qw, kw)
        mask_axis_2 = tf.expand_dims(tf.expand_dims(mask, 1), 3)
        logits = logits * mask_axis_2 + (1 - mask_axis_2) * -1e12
        mask_axis_3 = tf.expand_dims(tf.expand_dims(mask, 1), 1)
        logits = logits * mask_axis_3 + (1 - mask_axis_3) * -1e12
        # 排除下三角
        mask = tf.linalg.band_part(tf.ones_like(logits), 0, -1)
        logits = logits - (1 - mask) * 1e12
        flatten_logits = tf.reshape(
            logits / self.head_size ** 0.5, [input_shape[0], self.heads, -1]
        )
        return tf.RaggedTensor.from_tensor(flatten_logits)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape([input_shape[0], self.heads, input_shape[1] ** 2])

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "heads": self.heads,
            "head_size": self.head_size,
            "max_sequence_length": self.max_sequence_length,
        }