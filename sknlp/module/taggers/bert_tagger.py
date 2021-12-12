from __future__ import annotations
from typing import Any, Sequence, Optional

import tensorflow as tf
from tensorflow.keras.layers import Dropout

from sknlp.layers import BertPreprocessingLayer, BertAttentionMaskLayer
from sknlp.data import BertTaggingDataset
from sknlp.module.text2vec import Bert2vec
from .deep_tagger import DeepTagger


class BertTagger(DeepTagger):
    dataset_class = BertTaggingDataset

    def __init__(
        self,
        classes: Sequence[str],
        output_format: str = "global_pointer",
        global_pointer_head_size: int = 64,
        crf_learning_rate_multiplier: float = 1.0,
        max_sequence_length: int = 120,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 256,
        fc_activation: str = "tanh",
        text2vec: Optional[Bert2vec] = None,
        text_normalization: dict[str, str] = {"letter_case": "lowercase"},
        **kwargs
    ) -> None:
        super().__init__(
            classes,
            add_start_end_tag=True,
            output_format=output_format,
            global_pointer_head_size=global_pointer_head_size,
            crf_learning_rate_multiplier=crf_learning_rate_multiplier,
            algorithm="bert",
            num_fc_layers=num_fc_layers,
            fc_hidden_size=fc_hidden_size,
            fc_activation=fc_activation,
            max_sequence_length=max_sequence_length,
            text2vec=text2vec,
            text_normalization=text_normalization,
            **kwargs
        )
        self.inputs = [
            tf.keras.Input(shape=(None,), dtype=tf.int64, name="token_ids"),
            tf.keras.Input(shape=(None,), dtype=tf.int64, name="type_ids"),
        ]

        if self.output_format == "bio":
            self.inputs.append(
                tf.keras.Input(shape=(None,), dtype=tf.int32, name="tag_id")
            )

    def build_encoding_layer(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        if self.output_format == "bio":
            token_ids, type_ids, tag_ids = inputs
        else:
            token_ids, type_ids = inputs
        mask_layer = tf.keras.layers.Lambda(
            lambda ids: tf.not_equal(ids, 0), name="mask_layer"
        )
        mask = mask_layer(token_ids)

        outputs = [
            self.text2vec(
                [token_ids, type_ids, BertAttentionMaskLayer()([type_ids, mask])]
            )[1],
            mask,
        ]
        if self.output_format == "bio":
            outputs.append(tag_ids)
        return outputs

    def export(self, directory: str, name: str, version: str = "0") -> None:
        inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")
        preprocessing_layer = BertPreprocessingLayer(
            self.text2vec.vocab.sorted_tokens, return_offset=True
        )
        token_ids, type_ids, starts, ends = preprocessing_layer(inputs)
        fill_value = tf.float32.min
        if self.output_format == "bio":
            fill_value = tf.cast(0, tf.int32)
        predictions = self._inference_model([token_ids, type_ids]).to_tensor(fill_value)

        original_model = self._inference_model
        self._inference_model = tf.keras.Model(
            inputs=inputs, outputs=[predictions, starts, ends]
        )
        super(DeepTagger, self).export(directory, name, version=version)
        self._inference_model = original_model

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BertTagger":
        config.pop("add_start_end_tag")
        return super().from_config(config)