from __future__ import annotations
from typing import Callable, Optional, Any
import os
import tempfile
from enum import Enum
from collections import Counter
import logging

import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal

from official.nlp.bert.configs import BertConfig

from sknlp.vocab import Vocab
from sknlp.activations import gelu
from sknlp.layers import BertLayer
from sknlp.layers.utils import (
    convert_bert_checkpoint,
    convert_electra_checkpoint,
    convert_albert_checkpoint,
)
from .text2vec import Text2vec


def create_checkpoint_file(checkpoint: str) -> None:
    checkpoint_filename = os.path.join(checkpoint, "checkpoint")
    if os.path.exists(checkpoint_filename):
        return

    for filename in os.listdir(checkpoint):
        if "ckpt" in filename:
            model_checkpoint = filename[: filename.index("ckpt") + 4]
            with open(checkpoint_filename, "w") as f:
                f.write(f'model_checkpoint_path: "{model_checkpoint}"\n')
                f.write(f'all_model_checkpoint_paths: "{model_checkpoint}"')
            return


def get_bert_config_filename(
    checkpoint: str, config_filename: Optional[str] = None
) -> str:
    if config_filename is not None and config_filename.endswith("json"):
        return config_filename

    json_files: list[str] = [
        filename for filename in os.listdir(checkpoint) if filename.endswith("json")
    ]
    if not json_files:
        raise FileNotFoundError(f"{checkpoint}中缺少模型配置文件")
    if len(json_files) == 1:
        return json_files[0]
    else:
        for json_file in json_files:
            if "generator" not in json_file:
                return json_file


class BertFamily(Enum):
    BERT = 1
    ALBERT = 2
    ELECTRA = 3


class Bert2vec(Text2vec):
    def __init__(
        self,
        vocab: Vocab,
        segmenter: Optional[str] = None,
        hidden_size: int = 768,
        embedding_size: Optional[int] = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        sequence_length: Optional[int] = None,
        max_sequence_length: int = 512,
        type_vocab_size: int = 16,
        intermediate_size: int = 3072,
        activation: Callable = gelu,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        initializer: tf.keras.initializers.Initializer = TruncatedNormal(stddev=0.02),
        return_all_layer_outputs: bool = False,
        share_layer: bool = False,
        cls_pooling: bool = True,
        name: str = "bert2vec",
        **kwargs,
    ) -> None:
        if embedding_size is None:
            embedding_size = hidden_size
        super().__init__(
            vocab,
            segmenter=segmenter,
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
            embedding_size=embedding_size,
            name=name,
            **kwargs,
        )
        self.return_all_layer_outputs = return_all_layer_outputs
        self.bert_layer = BertLayer(
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
        attention_mask = tf.keras.Input(
            shape=(sequence_length, sequence_length),
            dtype=tf.float32,
            name="attention_mask",
        )
        logits_mask = tf.keras.Input(
            shape=(sequence_length,),
            dtype=tf.int64,
            name="logits_mask",
        )
        self.inputs = [token_ids, type_ids, attention_mask, logits_mask]
        self._model = tf.keras.Model(
            inputs=self.inputs, outputs=self.bert_layer(self.inputs), name=name
        )

    def __call__(
        self, inputs: list[tf.Tensor], logits_mask: Optional[tf.Tensor] = None
    ) -> list[tf.Tensor]:
        if logits_mask is None:
            inputs.append(tf.ones_like(inputs[0]))
        else:
            inputs.append(logits_mask)
        outputs = super().__call__(inputs)
        if not self.return_all_layer_outputs:
            return [outputs[0], outputs[1][-1], *outputs[2:]]
        return outputs

    def update_dropout(self, dropout: float) -> None:
        bert_layer = self._model.get_layer(self.name)
        bert_layer.embedding_dropout_layer.rate = dropout
        transformer_layers = []
        layer = getattr(bert_layer, "shared_layer", None)
        if layer:
            transformer_layers.append(layer)
        else:
            transformer_layers.extend(getattr(bert_layer, "transformer_layers", []))
        for layer in transformer_layers:
            layer._attention_dropout.rate = dropout
            layer._output_dropout.rate = dropout

    @classmethod
    def from_tfv1_checkpoint(
        cls,
        model_type: BertFamily,
        v1_checkpoint: str,
        config_filename: Optional[str] = None,
        cls_pooling: bool = True,
        sequence_length: Optional[int] = None,
        verbose: bool = False,
        name: str = "bert2vec",
    ):
        logger = logging.getLogger("sknlp-pretrain-converter")
        if verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        create_checkpoint_file(v1_checkpoint)
        config_filename = get_bert_config_filename(
            v1_checkpoint, config_filename=config_filename
        )
        config = BertConfig.from_json_file(os.path.join(v1_checkpoint, config_filename))
        with open(os.path.join(v1_checkpoint, "vocab.txt")) as f:
            token_list = f.read().strip("\n").split("\n")
            vocab = Vocab(
                Counter(token_list),
                pad_token=token_list[0],
                unk_token=token_list[100],
                bos_token=token_list[101],
                eos_token=token_list[102],
            )
        share_layer, cls_pooling = False, True
        convert_checkpoint = convert_bert_checkpoint
        if model_type is BertFamily.ALBERT:
            share_layer = True
            convert_checkpoint = convert_albert_checkpoint
        if model_type is BertFamily.ELECTRA:
            cls_pooling = False
            convert_checkpoint = convert_electra_checkpoint
        activation = tf.keras.activations.get(config.hidden_act)
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
            initializer=TruncatedNormal(stddev=config.initializer_range),
            share_layer=share_layer,
            cls_pooling=cls_pooling,
            name=name,
        )
        module.build()
        variable_mapping = convert_checkpoint(
            checkpoint_from_path=v1_checkpoint,
            num_heads=config.num_attention_heads,
            converted_root_name=name,
        )
        weight_value_pairs = []
        missing_variables = set()
        for variable in module._model.trainable_weights:
            name = variable.name.split(":")[0]
            if name in variable_mapping:
                weight_value_pairs.append((variable, variable_mapping[name]))
            else:
                missing_variables.add(variable.name)
        ignored_variables = set(variable_mapping.keys()) - {
            v.name.split(":")[0] for v in module._model.trainable_weights
        }
        if verbose:
            logger.info(f"Ignored variables: {ignored_variables}")
            logger.info(f"Missing variables: {missing_variables}")
        tf.keras.backend.batch_set_value(weight_value_pairs)
        return module

    def export(
        self,
        directory: str,
        name: str,
        version: str = "0",
        only_output_cls: bool = False,
    ) -> None:
        original_model = self._model
        if only_output_cls:
            self._model = tf.keras.Model(
                inputs=self._model.inputs,
                outputs=self._model.outputs[1],
                name="bert2vec",
            )
        super().export(directory, name, version)
        d = os.path.join(directory, name, version)
        self.save_vocab(d)
        self._model = original_model

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "return_all_layer_outputs": self.return_all_layer_outputs,
        }