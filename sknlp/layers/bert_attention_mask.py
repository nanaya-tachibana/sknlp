from __future__ import annotations
from typing import Optional, Any
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="sknlp")
class BertAttentionMaskLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        mask_mode: Optional[str] = None,
        dtype: tf.DType = tf.float32,
        name: str = "attention_mask",
        **kwargs
    ) -> None:
        super().__init__(dtype=dtype, name=name, **kwargs)
        self.mask_mode = mask_mode

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        type_ids, mask = inputs
        mask = tf.cast(mask, self.dtype)
        attention_mask = mask[..., None] * mask[:, None, :]
        if self.mask_mode == "unilm":
            pos = tf.cumsum(type_ids, axis=1)
            attention_mask *= tf.cast(pos[:, None, :] <= pos[..., None], self.dtype)
        return attention_mask

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "mask_mode": self.mask_mode}
