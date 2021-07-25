from __future__ import annotations
from typing import Any, Sequence, Optional

import tensorflow as tf
from tensorflow.keras.layers import Dropout

from sknlp.layers import BertCharPreprocessingLayer
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
        dropout: float = 0.5,
        text2vec: Optional[Bert2vec] = None,
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
            **kwargs
        )
        self.dropout = dropout
        if self.output_format == "bio":
            self.inputs = [
                tf.keras.Input(shape=(), dtype=tf.string, name="text_input"),
                tf.keras.Input(shape=(None,), dtype=tf.int32, name="tag_id"),
            ]
        else:
            self.inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")

    def build_encode_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.output_format == "bio":
            texts = inputs[0]
        else:
            texts = inputs
        token_ids = BertCharPreprocessingLayer(self.text2vec.vocab.sorted_tokens)(texts)

        mask = tf.keras.layers.Lambda(
            lambda x: tf.cast(x != 0, tf.int32), name="mask_layer"
        )(token_ids)
        if self.dropout:
            self.text2vec.update_dropout(dropout=self.dropout)
        embeddings, _ = self.text2vec(
            [token_ids, tf.zeros_like(token_ids, dtype=tf.int64)]
        )
        if self.dropout:
            noise_shape = (None, 1, self.text2vec.embedding_size)
            embeddings = Dropout(
                self.dropout,
                noise_shape=noise_shape,
                name="encoder_dropout",
            )(embeddings)

        if self.output_format == "bio":
            return embeddings, mask, inputs[1]
        else:
            return embeddings, mask

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "dropout": self.dropout}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BertTagger":
        config.pop("add_start_end_tag")
        return super().from_config(config)