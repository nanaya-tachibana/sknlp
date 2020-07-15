from typing import Optional, List, Dict, Any
import os
from collections import Counter

import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal
from official.modeling import activations

from sknlp.vocab import Vocab
from sknlp.layers.bert_layer import BertLayer, BertPreprocessingLayer
from sknlp.layers.bert_tokenization import BertTokenizationLayer

from .text2vec import Text2vec


class Bert2vec(Text2vec):

    def __init__(
        self,
        vocab: Vocab,
        segmenter: Optional[str] = None,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        sequence_length: int = 512,
        max_sequence_length: int = 512,
        type_vocab_size: int = 16,
        intermediate_size: int = 3072,
        activation: activations = activations.gelu,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        initializer: tf.keras.initializers.Initializer = TruncatedNormal(stddev=0.02),
        return_all_encoder_outputs: bool = False,
        only_output_cls = False,
        name: str = "bert2vec",
        **kwargs
    ) -> None:
        super().__init__(
            vocab,
            segmenter=segmenter,
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
            name=name,
            **kwargs
        )
        self._hidden_size = hidden_size
        self._only_output_cls = only_output_cls
        bert_preprocessing_layer = BertPreprocessingLayer(
            self.vocab.sorted_tokens[2:], sequence_length
        )
        bert_layer = BertLayer(
            len(self.vocab),
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
            return_all_encoder_outputs=return_all_encoder_outputs,
            name="bert_layer"
        )
        inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")
        if only_output_cls:
            _, outputs = bert_layer(bert_preprocessing_layer(inputs))
        else:
            outputs = bert_layer(bert_preprocessing_layer(inputs))
        self._model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

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
        if self._only_output_cls:
            return ["bert_layer"]
        return ["bert_layer", "bert_layer_1"]

    @property
    def output_types(self) -> List[str]:
        if self._only_output_cls:
            return ["float"]
        return ["float", "float"]

    @property
    def output_shapes(self) -> List[List[int]]:
        if self._only_output_cls:
            return [[-1, self.embedding_size]]
        sequence_length = self.sequence_length or -1
        return [[-1, sequence_length, self.embedding_size], [-1, self.embedding_size]]

    @property
    def embedding_size(self):
        return self._hidden_size

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "hidden_size": self.embedding_size}

    @classmethod
    def from_tfv1_checkpoint(
        cls, v1_checkpoint, only_output_cls=False, sequence_length=None
    ):
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
            return_all_encoder_outputs=bert_layer.return_all_encoder_outputs,
            only_output_cls=only_output_cls
        )
        inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")
        if only_output_cls:
            _, outputs = bert_layer(bert_preprocessing_layer(inputs))
        else:
            outputs = bert_layer(bert_preprocessing_layer(inputs))
        module._model = tf.keras.Model(inputs=inputs, outputs=outputs, name="bert2vec")
        return module

    def get_custom_objects(self) -> Dict[str, Any]:
        return {
            **super().get_custom_objects(),
            "BertTokenizationLayer": BertTokenizationLayer,
            "BertLayer": BertLayer,
            "BertPreprocessingLayer": BertPreprocessingLayer,
        }
