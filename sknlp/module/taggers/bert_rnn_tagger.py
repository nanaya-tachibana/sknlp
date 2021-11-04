from __future__ import annotations
from typing import Sequence, Optional

import tensorflow as tf

from sknlp.layers import BiLSTM
from sknlp.module.text2vec import Bert2vec
from .bert_tagger import BertTagger


class BertRNNTagger(BertTagger):
    def __init__(
        self,
        classes: Sequence[str],
        output_format: str = "global_pointer",
        global_pointer_head_size: int = 64,
        crf_learning_rate_multiplier: float = 1.0,
        rnn_learning_rate_multiplier: float = 1.0,
        max_sequence_length: int = 120,
        num_rnn_layers: int = 1,
        rnn_hidden_size: int = 512,
        rnn_recurrent_dropout: float = 0.0,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 256,
        fc_activation: str = "tanh",
        dropout: float = 0.5,
        text2vec: Optional[Bert2vec] = None,
        **kwargs
    ) -> None:
        super().__init__(
            classes,
            output_format=output_format,
            global_pointer_head_size=global_pointer_head_size,
            crf_learning_rate_multiplier=crf_learning_rate_multiplier,
            max_sequence_length=max_sequence_length,
            num_fc_layers=num_fc_layers,
            fc_hidden_size=fc_hidden_size,
            fc_activation=fc_activation,
            dropout=dropout,
            text2vec=text2vec,
            **kwargs
        )
        self.algorithm = "bert-rnn"
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_recurrent_dropout = rnn_recurrent_dropout
        self.rnn_learning_rate_multiplier = rnn_learning_rate_multiplier

    def build_intermediate_layer(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        encodings = inputs[0]
        mask = inputs[1]
        rnn_layer = BiLSTM(
            self.num_rnn_layers,
            self.rnn_hidden_size,
            dropout=self.dropout,
            recurrent_dropout=self.rnn_recurrent_dropout,
            return_sequences=True,
        )
        if self.rnn_learning_rate_multiplier != 1.0:
            self._layerwise_learning_rate_multiplier.append(
                (rnn_layer, self.rnn_learning_rate_multiplier)
            )
        encodings = rnn_layer(encodings, mask)
        if self.dropout:
            noise_shape = (None, 1, self.rnn_hidden_size * 2)
            encodings = tf.keras.layers.Dropout(
                self.dropout,
                noise_shape=noise_shape,
                name="encoding_dropout",
            )(encodings)
        return [encodings, mask, *inputs[2:]]