from typing import Optional, List, Dict, Any
import os
from collections import Counter

import tensorflow as tf
from official.modeling import activations

from sknlp.vocab import Vocab
from sknlp.layers.bert_layer import BertLayer, BertPreprocessingLayer

from .text2vec import Text2vec


class Bert2vec(Text2vec):

    def __init__(
        self,
        vocab,
        segmenter=None,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        sequence_length=512,
        max_sequence_length=None,
        type_vocab_size=16,
        intermediate_size=3072,
        activation=activations.gelu,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        return_all_encoder_outputs=False,
        name="bert2vec",
        **kwargs
    ):
        super().__init__(vocab, segmenter=segmenter, name=name, **kwargs)
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._num_attention_heads = num_attention_heads
        self._sequence_length = sequence_length
        self._max_sequence_length = max_sequence_length
        self._type_vocab_size = type_vocab_size
        self._intermediate_size = intermediate_size
        self._activation = activation
        self._dropout_rate = dropout_rate
        self._attention_dropout_rate = attention_dropout_rate
        self._initializer = initializer
        self._return_all_encoder_outputs = return_all_encoder_outputs

        bert_preprocessing_layer = BertPreprocessingLayer(
            self._vocab.sorted_tokens[2:], sequence_length
        )
        bert_layer = BertLayer(
            len(self._vocab),
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
            type_vocab_size=type_vocab_size,
            intermediate_size=intermediate_size,
            activation=activation,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            initializer=initializer,
            return_all_encoder_outputs=return_all_encoder_outputs
        )
        inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")
        _, cls = bert_layer(bert_preprocessing_layer(inputs))
        self._model = tf.keras.Model(inputs=inputs, outputs=cls, name=name)

    @property
    def input_names(self) -> List[str]:
        return ["text_input"]

    @property
    def input_types(self) -> List[str]:
        return ["string"]

    @property
    def input_shapes(self) -> List[List[int]]:
        return [[-1, 1]]

    @property
    def output_names(self) -> List[str]:
        return ["transformer_encoder"]

    @property
    def output_types(self) -> List[str]:
        return ["float"]

    @property
    def output_shapes(self) -> List[List[int]]:
        return [[-1, self.embedding_size]]

    @property
    def embedding_size(self):
        return self._hidden_size

    def get_config(self) -> Dict[str, Any]:
        return {
            **super().get_config(),
            "hidden_size": self._hidden_size,
            "num_layers": self._num_layers,
            "num_attention_heads": self._num_attention_heads,
            "sequence_length": self._sequence_length,
            "max_sequence_length": self._max_sequence_length,
            "type_vocab_size": self._type_vocab_size,
            "intermediate_size": self._intermediate_size,
            "attention_dropout_rate": self._attention_dropout_rate,
            "return_all_encoder_outputs": self._return_all_encoder_outputs
        }

    @classmethod
    def from_tfv1_checkpoint(cls, v1_checkpoint, sequence_length=100):
        bert_layer = BertLayer.from_tfv1_checkpoint(v1_checkpoint, sequence_length)
        with open(os.path.join(v1_checkpoint, "vocab.txt")) as f:
            vocab_list = f.read().split("\n")
            bert_preprocessing_layer = BertPreprocessingLayer(
                vocab_list[2:], sequence_length
            )
            vocab = Vocab(
                Counter(vocab_list), pad_token=vocab_list[0], unk_token=vocab_list[1]
            )

        module = cls(
            vocab,
            hidden_size=bert_layer.hidden_size,
            num_layers=bert_layer.num_layers,
            num_attention_heads=bert_layer.num_attention_heads,
            sequence_length=bert_layer.sequence_length,
            max_sequence_length=bert_layer.max_sequence_length,
            type_vocab_size=bert_layer.type_vocab_size,
            intermediate_size=bert_layer.intermediate_size,
            activation=bert_layer.activation,
            dropout_rate=bert_layer.dropout_rate,
            attention_dropout_rate=bert_layer.attention_dropout_rate,
            initializer=bert_layer.initializer,
            return_all_encoder_outputs=bert_layer.return_all_encoder_outputs
        )
        inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")
        _, cls = bert_layer(bert_preprocessing_layer(inputs))
        module._model = tf.keras.Model(inputs=inputs, outputs=cls, name="bert2vec")
        return module

    @classmethod
    def _filter_config(cls, config):
        config = super()._filter_config(config)
        config.pop("segmenter", None)
        return config

    def get_custom_objects(self) -> Dict[str, Any]:
        return {
            **super().get_custom_objects(),
            "BertLayer": BertLayer,
            "BertPreprocessingLayer": BertPreprocessingLayer,
        }
