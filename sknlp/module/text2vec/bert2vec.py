from __future__ import annotations
from typing import Callable, Optional, Any
import os
from enum import Enum

import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal

from sknlp.vocab import Vocab
from sknlp.activations import gelu
from sknlp.layers import (
    BertLayer,
    BertPreprocessingLayer,
    BertAttentionMaskLayer,
)
from sknlp.layers.utils import (
    BertCheckpointConverter,
    AlbertCheckpointConverter,
    ElectraCheckpointConverter,
)
from .text2vec import Text2vec
from .bert_config import BertConfig


class BertFamily(Enum):
    BERT = 1
    ALBERT = 2
    ELECTRA = 3


class ModelCheckpoint:
    def __init__(
        self,
        model_type: BertFamily,
        checkpoint_directory: str,
        config: BertConfig,
        vocab: Vocab,
        checkpoint_prefix: str = "bert_model.ckpt",
    ) -> None:
        self.directory = checkpoint_directory
        self.prefix = checkpoint_prefix
        self.config = config
        self.vocab = vocab
        share_layer, cls_pooling = False, True
        self.converter_class = BertCheckpointConverter
        if model_type is BertFamily.ALBERT:
            share_layer = True
            self.converter_class = AlbertCheckpointConverter
        if model_type is BertFamily.ELECTRA:
            cls_pooling = False
            self.converter_class = ElectraCheckpointConverter
        self.config.share_layer = share_layer
        self.config.cls_pooling = cls_pooling

    def load(self, weights: list[tf.Variable], name: str = "bert2vec") -> None:
        self.converter = self.converter_class(
            self.config.num_hidden_layers,
            self.config.num_attention_heads,
            converted_root_name=name,
        )
        self.converter.convert(self.directory, weights)

    def save(
        self,
        weights: list[tf.Variable],
        name: str = "bert2vec",
        config_filename: str = "bert_config.json",
        vocab_filename: str = "vocab.txt",
    ) -> None:
        self.converter = self.converter_class(
            self.config.num_hidden_layers,
            self.config.num_attention_heads,
            converted_root_name=name,
        )
        self.converter.invert(weights, self.directory, checkpoint_prefix=self.prefix)
        config_filename = os.path.join(self.directory, config_filename)
        vocab_filename = os.path.join(self.directory, vocab_filename)
        with open(config_filename, "w", encoding="UTF-8") as f:
            f.write(self.config.to_json_string())
        with open(vocab_filename, "w", encoding="UTF-8") as f:
            f.write("\n".join(self.vocab.sorted_tokens))

    @classmethod
    def search_config_file(cls, directory: str) -> str:
        json_files = [
            filename for filename in os.listdir(directory) if filename.endswith("json")
        ]
        if not json_files:
            raise FileNotFoundError(f"{directory}中缺少模型配置文件.")
        if len(json_files) == 1:
            return json_files[0]
        else:
            for json_file in json_files:
                if "generator" not in json_file:
                    return json_file

    @classmethod
    def create_checkpoint_file(cls, directory: str) -> None:
        checkpoint_filename = os.path.join(directory, "checkpoint")
        if os.path.exists(checkpoint_filename):
            return

        for filename in os.listdir(directory):
            if filename.endswith("index"):
                model_checkpoint = ".".join(filename.split(".")[:-1])
                with open(checkpoint_filename, "w", encoding="UTF-8") as f:
                    f.write(f'model_checkpoint_path: "{model_checkpoint}"\n')
                    f.write(f'all_model_checkpoint_paths: "{model_checkpoint}"')
                return
        else:
            raise FileNotFoundError("没有找到合法的ckpt文件.")

    @classmethod
    def from_checkpoint_file(
        cls,
        model_type: BertFamily,
        checkpoint_directory: str,
        config_filename: Optional[str] = None,
    ) -> "ModelCheckpoint":
        cls.create_checkpoint_file(checkpoint_directory)
        with open(
            os.path.join(checkpoint_directory, "vocab.txt"), encoding="UTF-8"
        ) as f:
            token_list = f.read().strip("\n").split("\n")
            vocab = Vocab(
                token_list,
                pad_token=token_list[0],
                unk_token=token_list[100],
                bos_token=token_list[101],
                eos_token=token_list[102],
            )
        if not config_filename:
            config_filename = cls.search_config_file(checkpoint_directory)
        config = BertConfig.from_json_file(
            os.path.join(checkpoint_directory, config_filename)
        )
        return cls(model_type, checkpoint_directory, config, vocab)


class Bert2vec(Text2vec):
    def __init__(
        self,
        vocab: Vocab,
        segmenter: Optional[str] = "bert",
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
        add_relationship_loss: bool = False,
        enable_recompute_grad: bool = False,
        model_type: BertFamily = BertFamily.BERT,
        name: str = "bert2vec",
        **kwargs,
    ) -> None:
        super().__init__(
            vocab,
            segmenter=segmenter,
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
            embedding_size=hidden_size,
            algorithm="bert",
            name=name,
            **kwargs,
        )
        self.model_type = model_type
        self.return_all_layer_outputs = return_all_layer_outputs
        self.add_relationship_loss = add_relationship_loss
        self.pretrain_layer = BertLayer(
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
            enable_recompute_grad=enable_recompute_grad,
            name=self.name,
        )
        self.inputs = [
            tf.keras.Input(shape=(None,), dtype=tf.int64, name="token_ids"),
            tf.keras.Input(shape=(None,), dtype=tf.int64, name="type_ids"),
            tf.keras.Input(
                shape=(sequence_length,), dtype=tf.int64, name="lm_logits_mask"
            ),
        ]
        if self.add_relationship_loss:
            self.inputs.insert(
                1, tf.keras.Input(shape=(), dtype=tf.string, name="relation_text_input")
            )

    def build_encoding_layer(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        token_ids, type_ids, logits_mask = inputs
        mask = tf.math.not_equal(token_ids, 0)
        return self.pretrain_layer(
            [
                token_ids,
                type_ids,
                BertAttentionMaskLayer()([type_ids, mask]),
                logits_mask,
            ]
        )

    def build_output_layer(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        return inputs[-1]

    def __call__(
        self, inputs: list[tf.Tensor], logits_mask: Optional[tf.Tensor] = None
    ) -> list[tf.Tensor]:
        if logits_mask is None:
            inputs.append(tf.ones_like(inputs[0]))
        else:
            inputs.append(logits_mask)
        outputs = self.pretrain_layer(inputs)
        if not self.return_all_layer_outputs:
            return [outputs[0], outputs[1][-1], *outputs[2:]]
        return outputs

    def update_dropout(
        self, dropout: float, attention_dropout: Optional[float] = None
    ) -> None:
        bert_layer = self.pretrain_layer
        bert_layer.embedding_dropout_layer.rate = dropout
        transformer_layers = []
        layer = getattr(bert_layer, "shared_layer", None)
        if layer:
            transformer_layers.append(layer)
        else:
            transformer_layers.extend(getattr(bert_layer, "transformer_layers", []))
        for layer in transformer_layers:
            if attention_dropout is not None:
                layer._attention_layer._dropout_layer.rate = attention_dropout
            layer._attention_dropout.rate = dropout
            layer._output_dropout.rate = dropout

    @classmethod
    def from_tfv1_checkpoint(
        cls,
        model_type: BertFamily,
        checkpoint_directory: str,
        config_filename: Optional[str] = None,
        sequence_length: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        attention_dropout_rate: Optional[float] = None,
        enable_recompute_grad: bool = False,
        name: str = "bert2vec",
    ):
        checkpoint = ModelCheckpoint.from_checkpoint_file(
            model_type,
            checkpoint_directory,
            config_filename=config_filename,
        )
        config = checkpoint.config
        activation = tf.keras.activations.get(config.hidden_act)
        dropout_rate = dropout_rate or config.hidden_dropout_prob
        attention_dropout_rate = (
            attention_dropout_rate or config.attention_probs_dropout_prob
        )
        module = cls(
            checkpoint.vocab,
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            activation=activation,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            sequence_length=sequence_length,
            max_sequence_length=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            initializer=TruncatedNormal(stddev=config.initializer_range),
            share_layer=config.share_layer,
            cls_pooling=config.cls_pooling,
            enable_recompute_grad=enable_recompute_grad,
            model_type=model_type,
            name=name,
        )
        module.build()
        checkpoint.load(module._model.trainable_weights, name=name)
        return module

    def to_tfv1_checkpoint(self, checkpoint_directory: str) -> None:
        self.save_bert_tfv1_checkpoint(
            self.model_type,
            checkpoint_directory,
            self.pretrain_layer,
            self.pretrain_layer.get_config(),
            self.vocab,
        )

    @classmethod
    def save_bert_tfv1_checkpoint(
        cls,
        model_type: BertFamily,
        checkpoint_directory: str,
        bert_layer: BertLayer,
        bert_layer_config: dict[str, Any],
        vocab: Vocab,
    ) -> None:
        hidden_act = bert_layer_config["activation"]
        if not isinstance(hidden_act, str):
            hidden_act = tf.keras.activations.serialize(hidden_act)
        config = BertConfig(
            bert_layer_config["vocab_size"],
            hidden_size=bert_layer_config["hidden_size"],
            num_hidden_layers=bert_layer_config["num_layers"],
            num_attention_heads=bert_layer_config["num_attention_heads"],
            intermediate_size=bert_layer_config["intermediate_size"],
            type_vocab_size=bert_layer_config["type_vocab_size"],
            hidden_act=hidden_act,
            hidden_dropout_prob=bert_layer_config["dropout_rate"],
            attention_probs_dropout_prob=bert_layer_config["attention_dropout_rate"],
            embedding_size=bert_layer_config["embedding_size"],
        )
        checkpoint = ModelCheckpoint(model_type, checkpoint_directory, config, vocab)
        checkpoint.save(bert_layer.trainable_weights, name=bert_layer_config["name"])

    def export(
        self,
        directory: str,
        name: str,
        version: str = "0",
        only_output_cls: bool = False,
    ) -> None:
        self.text_input = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")
        preprocessing_layer = BertPreprocessingLayer(self.vocab.sorted_tokens)
        attention_mask_layer = BertAttentionMaskLayer()
        token_ids, type_ids = preprocessing_layer(self.text_input)
        mask = tf.not_equal(token_ids, 0)
        attention_mask = attention_mask_layer([type_ids, mask])
        outputs = self.pretrain_layer(
            [token_ids, type_ids, attention_mask, tf.ones_like(token_ids)]
        )
        if only_output_cls:
            self._inference_model = tf.keras.Model(
                inputs=self.text_input,
                outputs=outputs[0],
            )
        else:
            self._inference_model = tf.keras.Model(
                inputs=self.text_input,
                outputs=[outputs[0], outputs[1][-1]],
            )
        super().export(directory, name, version)
        d = os.path.join(directory, name, version)
        self.save_vocab(d)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "return_all_layer_outputs": self.return_all_layer_outputs,
        }