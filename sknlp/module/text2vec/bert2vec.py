from typing import Optional, Dict, Any, Callable
import os
import tempfile
from enum import Enum
from collections import Counter

import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal
from official.modeling import activations
from official.nlp.bert.configs import BertConfig

from sknlp.vocab import Vocab
from sknlp.layers import BertLayer
from sknlp.layers.utils import (
    convert_bert_checkpoint,
    convert_electra_checkpoint,
    convert_albert_checkpoint,
)
from .text2vec import Text2vec


class BertFamily(Enum):
    BERT = 1
    ALBERT = 2
    ELECTRA = 3


def get_activation(activation_string: str) -> Callable:
    if activation_string == "gelu":
        return activations.gelu
    else:
        return tf.keras.activations.get(activation_string)


class Bert2vec(Text2vec):
    def __init__(
        self,
        vocab: Vocab,
        segmenter: Optional[str] = None,
        hidden_size: int = 768,
        embedding_size: int = 768,
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
        share_layer: bool = False,
        cls_pooling: bool = True,
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
        if embedding_size is None:
            embedding_size = hidden_size
        self._embedding_size = embedding_size
        bert_layer = BertLayer(
            len(self.vocab.sorted_tokens),
            embedding_size=embedding_size,
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
            share_layer=share_layer,
            cls_pooling=cls_pooling,
            name=name,
        )
        token_ids = tf.keras.Input(
            shape=(sequence_length,), dtype=tf.int64, name="input_token_ids"
        )
        type_ids = tf.keras.Input(
            shape=(sequence_length,), dtype=tf.int64, name="input_type_ids"
        )
        outputs = bert_layer([token_ids, type_ids])
        self._model = tf.keras.Model(
            inputs=[token_ids, type_ids], outputs=outputs, name=name
        )

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._model(inputs)

    @property
    def embedding_size(self):
        return self._embedding_size

    @classmethod
    def from_tfv1_checkpoint(
        cls,
        model_type: BertFamily,
        v1_checkpoint: str,
        config_filename: str = "bert_config.json",
        cls_pooling: bool = True,
        sequence_length: Optional[int] = None,
        name: str = "bert",
    ):
        config = BertConfig.from_json_file(os.path.join(v1_checkpoint, config_filename))
        with open(os.path.join(v1_checkpoint, "vocab.txt")) as f:
            token_list = f.read().strip("\n").split("\n")
            vocab = Vocab(
                Counter(token_list), pad_token=token_list[0], unk_token=token_list[1]
            )
        share_layer, cls_pooling = False, True
        convert_checkpoint = convert_bert_checkpoint
        if model_type is BertFamily.ALBERT:
            share_layer = True
            convert_checkpoint = convert_albert_checkpoint
        if model_type is BertFamily.ELECTRA:
            cls_pooling = False
            convert_checkpoint = convert_electra_checkpoint
        activation = get_activation(config.hidden_act)
        module = cls(
            vocab,
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            activation=activation,
            dropout_rate=config.hidden_dropout_prob,
            attention_dropout_rate=config.attention_probs_dropout_prob,
            sequence_length=sequence_length,
            max_sequence_length=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            initializer=tf.keras.initializers.TruncatedNormal(
                stddev=config.initializer_range
            ),
            share_layer=share_layer,
            cls_pooling=cls_pooling,
            name=name,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temporary_checkpoint = os.path.join(temp_dir, "ckpt")
            convert_checkpoint(
                checkpoint_from_path=v1_checkpoint,
                checkpoint_to_path=temporary_checkpoint,
                num_heads=config.num_attention_heads,
                converted_root_name=name,
            )
            module._model.load_weights(
                temporary_checkpoint
            ).assert_existing_objects_matched()
        return module

    def export(
        self,
        directory: str,
        name: str,
        version: str = "0",
        only_output_cls: bool = False,
    ) -> None:
        if only_output_cls:
            self._model = tf.keras.Model(
                inputs=self._model.inputs,
                outputs=self._model.outputs[1:],
                name="bert2vec",
            )
        super().export(directory, name, version)
        d = os.path.join(directory, name, version)
        self.save_vocab(d)

    def get_custom_objects(self) -> Dict[str, Any]:
        return {**super().get_custom_objects(), "BertLayer": BertLayer}