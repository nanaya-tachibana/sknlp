from typing import Optional, List, Dict, Any
import os
from collections import Counter

import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal
from official.modeling import activations

from sknlp.vocab import Vocab
from sknlp.layers import BertLayer, BertPreprocessingLayer, AlbertLayer

from .text2vec import Text2vec


class Bert2vec(Text2vec):

    def __init__(
        self,
        vocab: Vocab,
        segmenter: Optional[str] = None,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        sequence_length: Optional[int] = None,
        max_sequence_length: int = 512,
        type_vocab_size: int = 16,
        intermediate_size: int = 3072,
        activation: activations = activations.gelu,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        initializer: tf.keras.initializers.Initializer = TruncatedNormal(stddev=0.02),
        return_all_encoder_outputs: bool = False,
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
        self._only_output_cls = False
        bert_preprocessing_layer = BertPreprocessingLayer(self.vocab.sorted_tokens)
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
        inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")
        outputs = bert_layer(bert_preprocessing_layer(inputs))
        self._model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
        self.bert_layer = bert_layer

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._model(inputs)

    @property
    def embedding_size(self):
        return self._hidden_size

    @classmethod
    def from_tfv1_checkpoint(
        cls,
        v1_checkpoint: str,
        config_filename: str = "bert_config.json",
        sequence_length: Optional[int] = None
    ):
        bert_layer = BertLayer.from_tfv1_checkpoint(
            v1_checkpoint,
            config_filename=config_filename,
            sequence_length=sequence_length
        )
        with open(os.path.join(v1_checkpoint, "vocab.txt")) as f:
            token_list = f.read().split("\n")
            bert_preprocessing_layer = BertPreprocessingLayer(token_list)
            vocab = Vocab(
                Counter(token_list), pad_token=token_list[0], unk_token=token_list[1]
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
        )
        inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")
        outputs = bert_layer(bert_preprocessing_layer(inputs))
        module._model = tf.keras.Model(inputs=inputs, outputs=outputs, name="bert2vec")
        return module

    def export(
        self,
        directory: str,
        name: str,
        version: str = "0",
        only_output_cls: bool = False
    ) -> None:
        if only_output_cls:
            self._only_output_cls = True
            self._model = tf.keras.Model(
                inputs=self._model.inputs,
                outputs=self._model.outputs[1:],
                name="bert2vec"
            )
        super().export(directory, name, version)
        d = os.path.join(directory, name, version)
        self.save_vocab(d)

    def get_custom_objects(self) -> Dict[str, Any]:
        return {
            **super().get_custom_objects(),
            "BertLayer": BertLayer,
            "BertPreprocessingLayer": BertPreprocessingLayer,
        }



class Albert2vec(Text2vec):

    def __init__(
        self,
        vocab: Vocab,
        segmenter: Optional[str] = None,
        embedding_size: int = 128,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        sequence_length: Optional[int] = None,
        max_sequence_length: int = 512,
        type_vocab_size: int = 16,
        intermediate_size: int = 3072,
        activation: activations = activations.gelu,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        initializer: tf.keras.initializers.Initializer = TruncatedNormal(stddev=0.02),
        name: str = "albert2vec",
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
        self._embedding_size = embedding_size
        self._only_output_cls = False
        bert_preprocessing_layer = BertPreprocessingLayer(self.vocab.sorted_tokens)
        bert_layer = AlbertLayer(
            len(self.vocab),
            hidden_size=hidden_size,
            embedding_size=embedding_size,
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
            name="albert_layer"
        )
        inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")
        outputs = bert_layer(bert_preprocessing_layer(inputs))
        self._model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
        self.bert_layer = bert_layer

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._model(inputs)

    @property
    def embedding_size(self):
        return self._embedding_size

    @classmethod
    def from_tfv1_checkpoint(
        cls,
        v1_checkpoint: str,
        config_filename: str = "albert_config.json",
        sequence_length: Optional[int] = None
    ):
        bert_layer = AlbertLayer.from_tfv1_checkpoint(
            v1_checkpoint,
            config_filename=config_filename,
            sequence_length=sequence_length
        )
        with open(os.path.join(v1_checkpoint, "vocab.txt")) as f:
            token_list = f.read().split("\n")
            bert_preprocessing_layer = BertPreprocessingLayer(token_list)
            vocab = Vocab(
                Counter(token_list), pad_token=token_list[0], unk_token=token_list[1]
            )

        module = cls(
            vocab,
            embedding_size=bert_layer.embedding_size,
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
        )
        inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")
        outputs = bert_layer(bert_preprocessing_layer(inputs))
        module._model = tf.keras.Model(inputs=inputs, outputs=outputs, name="albert2vec")
        return module

    def export(
        self,
        directory: str,
        name: str,
        version: str = "0",
        only_output_cls: bool = False
    ) -> None:
        if only_output_cls:
            self._only_output_cls = True
            self._model = tf.keras.Model(
                inputs=self._model.inputs,
                outputs=self._model.outputs[1:],
                name="albert2vec"
            )
        super().export(directory, name, version)
        d = os.path.join(directory, name, version)
        self.save_vocab(d)

    def get_custom_objects(self) -> Dict[str, Any]:
        return {
            **super().get_custom_objects(),
            "AlbertLayer": AlbertLayer,
            "BertPreprocessingLayer": BertPreprocessingLayer,
        }
