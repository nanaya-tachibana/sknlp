from typing import Any, Dict, Sequence, Optional

import tensorflow as tf
from tensorflow.keras.layers import Dropout

from sknlp.layers import BertPairPreprocessingLayer
from sknlp.data import BertSimilarityDataset
from sknlp.module.text2vec import Bert2vec
from .deep_discriminator import DeepDiscriminator


class BertDiscriminator(DeepDiscriminator):
    dataset_class = BertSimilarityDataset

    def __init__(
        self,
        classes: Sequence[str] = ("相似度",),
        max_sequence_length: int = 120,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 256,
        fc_activation: str = "tanh",
        dropout: float = 0.5,
        text2vec: Optional[Bert2vec] = None,
        **kwargs
    ) -> None:
        super().__init__(
            classes=classes,
            algorithm="bert",
            max_sequence_length=max_sequence_length,
            text2vec=text2vec,
            **kwargs
        )
        self.num_fc_layers = num_fc_layers
        self.fc_hidden_size = fc_hidden_size
        self.fc_activation = fc_activation
        self.dropout = dropout
        self.inputs = [
            tf.keras.Input(shape=(), dtype=tf.string, name="text_input"),
            tf.keras.Input(shape=(), dtype=tf.string, name="context_input"),
        ]

    def build_encode_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        preprocessing_layer = BertPairPreprocessingLayer(
            self.text2vec.vocab.sorted_tokens
        )
        token_ids, type_ids = preprocessing_layer(inputs)
        if self.dropout:
            self.text2vec.update_dropout(dropout=self.dropout)
        _, cls = self.text2vec([token_ids, type_ids])
        if self.dropout:
            return Dropout(self.dropout, name="embedding_dropout")(cls)
        return cls

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "dropout": self.dropout}