from __future__ import annotations
from typing import Optional, Any

import tensorflow as tf

from sknlp.data import BertGenerationDataset
from sknlp.layers import (
    BertPreprocessingLayer,
    BertAttentionMaskLayer,
    BertLMLossLayer,
    BertBeamSearchDecoder,
)
from sknlp.module.text2vec import Text2vec
from .deep_generator import DeepGenerator


@tf.keras.utils.register_keras_serializable(package="sknlp")
class TensorAppend(tf.keras.layers.Layer):
    def __init__(self, value: int | float, axis: int = -1, **kwargs) -> None:
        self.value = value
        self.axis = axis
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        ragged_inputs = tf.ragged.boolean_mask(inputs, mask)
        value = tf.cast(self.value, ragged_inputs.dtype)
        paddings = tf.fill((tf.shape(inputs)[0], 1), value)
        return tf.concat([ragged_inputs, paddings], -1).to_tensor()

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "value": self.value, "axis": self.axis}


class BertGenerator(DeepGenerator):
    dataset_class = BertGenerationDataset

    def __init__(
        self,
        max_sequence_length: Optional[int] = None,
        beam_width: int = 3,
        text2vec: Optional[Text2vec] = None,
        text_normalization: dict[str, str] = {"letter_case": "lowercase"},
        **kwargs
    ):
        super().__init__(
            max_sequence_length=max_sequence_length,
            beam_width=beam_width,
            text2vec=text2vec,
            algorithm="bert",
            text_normalization=text_normalization,
            **kwargs
        )
        self.inputs = [
            tf.keras.Input(shape=(None,), dtype=tf.int64, name="token_ids"),
            tf.keras.Input(shape=(None,), dtype=tf.int64, name="type_ids"),
        ]

    def build_encoding_layer(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        token_ids, type_ids = inputs
        mask = tf.math.not_equal(token_ids, 0)

        logits_mask = tf.roll(type_ids, -1, 1)
        logits = self.text2vec(
            [
                token_ids,
                type_ids,
                BertAttentionMaskLayer(mask_mode="unilm")([type_ids, mask]),
            ],
            logits_mask=logits_mask,
        )[2]
        return logits, token_ids, type_ids

    def build_output_layer(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        logits, token_ids, type_ids = inputs
        masked_token_ids = tf.ragged.boolean_mask(
            token_ids, tf.cast(type_ids, tf.bool)
        ).to_tensor()
        sequence_mask = tf.sequence_mask(tf.reduce_sum(type_ids, axis=1))
        return BertLMLossLayer()([masked_token_ids, logits, sequence_mask])

    def build_inference_model(self) -> tf.keras.Model:
        sep_id = self.vocab[self.vocab.eos]
        decoder = BertBeamSearchDecoder(
            self._model.get_layer("bert2vec").get_config(),
            self.beam_width,
            sep_id,
            maximum_iterations=self.maximum_iterations,
            parallel_iterations=self.parallel_iterations,
        )
        decoder(
            [
                tf.constant([[101, 567, 45, 342, 102, 102]]),
                tf.constant([[0, 0, 0, 0, 0, 1]]),
            ],
            tf.constant([[0.0]]),
        )
        decoder.set_weights(self._model.get_layer("bert2vec").get_weights())

        token_ids, type_ids = self._model.inputs
        mask = tf.not_equal(token_ids, 0)
        token_ids = TensorAppend(sep_id, -1)(token_ids, mask)
        type_ids = TensorAppend(1, -1)(type_ids, mask)
        state = tf.zeros_like(token_ids[..., 0, None], dtype=tf.float32)
        predicted_ids = decoder([token_ids, type_ids], state)[0].predicted_ids
        ragged_predicted_ids = tf.RaggedTensor.from_tensor(predicted_ids[..., 0])
        self._inference_model = tf.keras.Model(
            inputs=self._model.inputs, outputs=ragged_predicted_ids
        )

    def export(self, directory: str, name: str, version: str = "0") -> None:
        original_model = self._inference_model
        inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")
        preprocessing_layer = BertPreprocessingLayer(self.text2vec.vocab.sorted_tokens)
        token_ids, type_ids = preprocessing_layer([inputs, ""])
        outputs = self._inference_model([token_ids, type_ids])
        self._inference_model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs.to_tensor(),
        )
        super().export(directory, name, version=version)
        self._inference_model = original_model