from __future__ import annotations
from typing import Optional, Any, Union

import tensorflow as tf

from sknlp.data import BertGenerationDataset
from sknlp.layers import (
    BertPreprocessingLayer,
    BertAttentionMaskLayer,
    BertLMLossLayer,
    BertBeamSearchDecoder,
    BertDecodeCell,
)
from sknlp.module.text2vec import Text2vec
from .deep_generator import DeepGenerator


class BertGenerator(DeepGenerator):
    dataset_class = BertGenerationDataset

    def __init__(
        self,
        max_sequence_length: Optional[int] = None,
        beam_width: int = 3,
        text2vec: Optional[Text2vec] = None,
        dropout: float = 0.5,
        **kwargs
    ):
        self.dropout = dropout
        self.inputs = [
            tf.keras.Input(shape=(), dtype=tf.string, name="text_input"),
            tf.keras.Input(shape=(), dtype=tf.string, name="target_input"),
        ]
        self.maximum_iterations = kwargs.pop("maximum_iterations", 50)
        self.parallel_iterations = kwargs.pop("parallel_iterations", 1)
        super().__init__(
            max_sequence_length=max_sequence_length,
            beam_width=beam_width,
            text2vec=text2vec,
            **kwargs
        )

    def build_preprocessing_layer(
        self, inputs: Union[tf.Tensor, list[tf.Tensor]]
    ) -> Union[tf.Tensor, list[tf.Tensor]]:
        preprocessing_layer = BertPreprocessingLayer(self.text2vec.vocab.sorted_tokens)
        return preprocessing_layer(inputs)

    def build_encoding_layer(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        if self.dropout:
            self.text2vec.update_dropout(dropout=self.dropout)
        token_ids, type_ids = inputs
        mask = tf.math.not_equal(token_ids, 0)

        logits_mask = tf.roll(type_ids, -1, 1)
        _, _, logits = self.text2vec(
            [
                token_ids,
                type_ids,
                BertAttentionMaskLayer(mask_mode="unilm")([type_ids, mask]),
            ],
            logits_mask=logits_mask,
        )
        return logits, token_ids, type_ids

    def build_output_layer(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        logits, token_ids, type_ids = inputs
        masked_token_ids = tf.ragged.boolean_mask(
            token_ids, tf.cast(type_ids, tf.bool)
        ).to_tensor()
        sequence_mask = tf.sequence_mask(tf.reduce_sum(type_ids, axis=1))
        return BertLMLossLayer()([masked_token_ids, logits, sequence_mask])

    def build_inference_model(self) -> tf.keras.Model:
        cell = BertDecodeCell(self._model, len(self.text2vec.vocab))
        decoder = BertBeamSearchDecoder(
            cell,
            self.beam_width,
            self.text2vec.vocab["[SEP]"],
            maximum_iterations=self.maximum_iterations,
            parallel_iterations=self.parallel_iterations,
        )
        tokenizer = BertPreprocessingLayer(self.text2vec.vocab.sorted_tokens)
        text_input = self._model.inputs[0]
        token_ids, type_ids = tokenizer([text_input, tf.fill(tf.shape(text_input), "")])
        state = tf.zeros((tf.shape(token_ids)[0], cell.state_size))
        predicted_ids = decoder([token_ids, type_ids], state)[0].predicted_ids
        return tf.keras.Model(inputs=text_input, outputs=predicted_ids[..., 0])

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "dropout": self.dropout}