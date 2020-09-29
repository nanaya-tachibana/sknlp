from typing import List, Dict, Any, Optional, Callable
import os
import tempfile

import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import InputSpec

from official.modeling import activations
from official.nlp.modeling import layers
from official.nlp.bert.configs import BertConfig
from official.nlp.albert.configs import AlbertConfig
import tensorflow_text as tftext

from .utils import (
    convert_bert_checkpoint, create_bert_model,
    convert_albert_checkpoint, create_albert_model,
)


def get_activation(activation_string: str) -> Callable:
    if activation_string == "gelu":
        return activations.gelu
    else:
        return tf.keras.activations.get(activation_string)


class BertPreprocessingLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        vocab: List[str] = None,
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        **kwargs
    ) -> None:
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.tokenizer = tftext.BertTokenizer(
            tf.lookup.StaticVocabularyTable(
                tf.lookup.KeyValueTensorInitializer(
                    vocab,
                    list(range(len(vocab))),
                    key_dtype=tf.string,
                    value_dtype=tf.int64
                ),
                1
            )
        )
        for i, token in enumerate(vocab):
            if token == cls_token:
                self.cls_id = i
            if token == sep_token:
                self.sep_id = i
        self.vocab = vocab
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        tokens = self.tokenizer.tokenize(inputs)
        flatten_tokens = tokens.merge_dims(-2, -1)
        cls_tokens = tf.reshape(
            tf.tile(tf.constant([self.cls_id], dtype=tf.int64), [tokens.nrows()]),
            [tokens.nrows(), 1]
        )
        sep_tokens = tf.reshape(
            tf.tile(tf.constant([self.sep_id], dtype=tf.int64), [tokens.nrows()]),
            [tokens.nrows(), 1]
        )
        ids = tf.concat([cls_tokens, flatten_tokens, sep_tokens], axis=1)
        return ids.to_tensor(0)

    def get_config(self) -> Dict[str, Any]:
        return {
            **super().get_config(),
            "vocab": self.vocab,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token
        }


class BertLayer(tf.keras.layers.Layer):
    """
    copy from https://github.com/tensorflow/models/blob/v2.2.0/official/nlp/modeling/networks/transformer_encoder.py.
    """
    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        sequence_length: int = None,
        max_sequence_length: int = 512,
        type_vocab_size: int = 16,
        intermediate_size: int = 3072,
        activation: activations = activations.gelu,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        initializer: tf.keras.initializers.Initializer = TruncatedNormal(stddev=0.02),
        return_all_encoder_outputs: bool = False,
        name: str = "bert_layer",
        **kwargs
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.sequence_length = sequence_length
        self.max_sequence_length = max_sequence_length
        self.type_vocab_size = type_vocab_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.initializer = initializer
        self.return_all_encoder_outputs = return_all_encoder_outputs
        self.input_spec = InputSpec(ndim=2, shape=(None, self.sequence_length))

        self.embedding_layer = layers.OnDeviceEmbedding(
            vocab_size,
            hidden_size,
            initializer=initializer,
            name="word_embeddings"
        )
        # Always uses dynamic slicing for simplicity.
        self.position_embedding_layer = layers.PositionEmbedding(
            initializer=initializer,
            use_dynamic_slicing=True,
            max_sequence_length=max_sequence_length,
            name="position_embeddings"
        )
        self.type_embedding_layer = layers.OnDeviceEmbedding(
            type_vocab_size,
            hidden_size,
            initializer=initializer,
            use_one_hot=True,
            name="type_embeddings"
        )
        self.normalize_layer = tf.keras.layers.LayerNormalization(
            name="embeddings/layer_norm",
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = layers.Transformer(
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                intermediate_activation=activation,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                kernel_initializer=initializer,
                name="transformer/layer_%d" % i
            )
            self.transformer_layers.append(layer)

        self.cls_output_layer = tf.keras.layers.Dense(
            units=hidden_size,
            activation="tanh",
            kernel_initializer=initializer,
            name="pooler_transform"
        )

        super().__init__(name=name, **kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        input_masks = inputs != 0
        input_types = tf.zeros_like(inputs, name="input_type_ids")

        word_embeddings = self.embedding_layer(inputs)
        position_embeddings = self.position_embedding_layer(word_embeddings)
        type_embeddings = self.type_embedding_layer(input_types)
        embeddings = word_embeddings + position_embeddings + type_embeddings
        embeddings = self.normalize_layer(embeddings)
        embeddings = tf.keras.layers.Dropout(rate=self.dropout_rate)(embeddings)

        data = embeddings
        attention_mask = layers.SelfAttentionMask()([data, input_masks])
        encoder_outputs = []
        for layer in self.transformer_layers:
            data = layer([data, attention_mask])
            encoder_outputs.append(data)

        first_token_tensor = tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x[:, 0:1, :], axis=1)
        )(encoder_outputs[-1])
        cls_output = self.cls_output_layer(first_token_tensor)
        if self.return_all_encoder_outputs:
            return [encoder_outputs, cls_output]
        else:
            return [encoder_outputs[-1], cls_output]

    def get_config(self):
        return {
            **super().get_config(),
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "sequence_length": self.sequence_length,
            "max_sequence_length": self.max_sequence_length,
            "type_vocab_size": self.type_vocab_size,
            "intermediate_size": self.intermediate_size,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "initializer": self.initializer,
            "return_all_encoder_outputs": self.return_all_encoder_outputs
        }

    @classmethod
    def from_tfv1_checkpoint(
        cls,
        v1_checkpoint: str,
        config_filename: str = "bert_config.json",
        sequence_length: Optional[int] = None
    ) -> None:
        config = BertConfig.from_json_file(
            os.path.join(v1_checkpoint, config_filename)
        )
        if config.hidden_act == "gelu":
            activation = activations.gelu
        else:
            activation = tf.keras.activations.get(config.hidden_act)

        layer = cls(
            vocab_size=config.vocab_size,
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
            )
        )
        bert_network = create_bert_model(config)
        with tempfile.TemporaryDirectory() as temp_dir:
            temporary_checkpoint = os.path.join(temp_dir, "ckpt")
            convert_bert_checkpoint(
                checkpoint_from_path=v1_checkpoint,
                checkpoint_to_path=temporary_checkpoint,
                num_heads=config.num_attention_heads
            )
            bert_network.load_weights(
                temporary_checkpoint
            ).assert_existing_objects_matched()

            with open(os.path.join(v1_checkpoint, "vocab.txt")) as f:
                vocab = f.read().split("\n")
                bert_preprocessing_layer = BertPreprocessingLayer(vocab)

            layer(bert_preprocessing_layer(tf.constant(["是的", "不是的"])))
            layer.embedding_layer.set_weights(
                bert_network.get_layer("word_embeddings").get_weights()
            )
            layer.position_embedding_layer.set_weights(
                bert_network.get_layer("position_embedding").get_weights()
            )
            layer.type_embedding_layer.set_weights(
                bert_network.get_layer("type_embeddings").get_weights()
            )
            layer.normalize_layer.set_weights(
                bert_network.get_layer("embeddings/layer_norm").get_weights()
            )
            for i in range(len(layer.transformer_layers)):
                layer.transformer_layers[i].set_weights(
                    bert_network.get_layer("transformer/layer_%d" % i).get_weights()
                )
            layer.cls_output_layer.set_weights(
                bert_network.get_layer("pooler_transform").get_weights()
            )
            return layer



class AlbertLayer(tf.keras.layers.Layer):
    """
    copy from https://github.com/tensorflow/models/blob/v2.3.0/official/nlp/modeling/networks/albert_transformer_encoder.py
    """
    def __init__(
        self,
        vocab_size: int = 100,
        embedding_size: int = 128,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        sequence_length: int = None,
        max_sequence_length: int = 512,
        type_vocab_size: int = 16,
        intermediate_size: int = 3072,
        activation: activations = activations.gelu,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        initializer: tf.keras.initializers.Initializer = TruncatedNormal(stddev=0.02),
        name: str = "albert_layer",
        **kwargs
    ) -> None:
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.sequence_length = sequence_length
        self.max_sequence_length = max_sequence_length
        self.type_vocab_size = type_vocab_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.initializer = initializer
        self.input_spec = InputSpec(ndim=2, shape=(None, self.sequence_length))

        self.embedding_layer = layers.OnDeviceEmbedding(
            vocab_size,
            embedding_size,
            initializer=initializer,
            name="word_embeddings"
        )
        self.embedding_projection_layer = tf.keras.layers.experimental.EinsumDense(
            "...x,xy->...y",
            output_shape=hidden_size,
            bias_axes="y",
            kernel_initializer=initializer,
            name="embedding_projection"
        )

        # Always uses dynamic slicing for simplicity.
        self.position_embedding_layer = layers.PositionEmbedding(
            initializer=initializer,
            use_dynamic_slicing=True,
            max_sequence_length=max_sequence_length,
            name="position_embeddings"
        )
        self.type_embedding_layer = layers.OnDeviceEmbedding(
            type_vocab_size,
            embedding_size,
            initializer=initializer,
            use_one_hot=True,
            name="type_embeddings"
        )
        self.normalize_layer = tf.keras.layers.LayerNormalization(
            name="embeddings/layer_norm",
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32
        )
        self.shared_layer = layers.Transformer(
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            intermediate_activation=activation,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            kernel_initializer=initializer,
            name="transformer"
        )
        self.cls_output_layer = tf.keras.layers.Dense(
            units=hidden_size,
            activation="tanh",
            kernel_initializer=initializer,
            name="pooler_transform"
        )

        super().__init__(name=name, **kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        input_masks = inputs != 0
        input_types = tf.zeros_like(inputs, name="input_type_ids")

        word_embeddings = self.embedding_layer(inputs)
        position_embeddings = self.position_embedding_layer(word_embeddings)
        type_embeddings = self.type_embedding_layer(input_types)
        embeddings = word_embeddings + position_embeddings + type_embeddings
        embeddings = self.normalize_layer(embeddings)
        embeddings = tf.keras.layers.Dropout(rate=self.dropout_rate)(embeddings)

        data = self.embedding_projection_layer(embeddings)
        attention_mask = layers.SelfAttentionMask()([data, input_masks])
        for _ in range(self.num_layers):
            data = self.shared_layer([data, attention_mask])

        first_token_tensor = tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x[:, 0:1, :], axis=1)
        )(data)
        cls_output = self.cls_output_layer(first_token_tensor)
        return [data, cls_output]

    def get_config(self):
        return {
            **super().get_config(),
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "sequence_length": self.sequence_length,
            "max_sequence_length": self.max_sequence_length,
            "type_vocab_size": self.type_vocab_size,
            "intermediate_size": self.intermediate_size,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "initializer": self.initializer,
        }

    @classmethod
    def from_tfv1_checkpoint(
        cls,
        v1_checkpoint: str,
        config_filename: str = "albert_config.json",
        sequence_length: Optional[int] = None
    ) -> None:
        config = AlbertConfig.from_json_file(
            os.path.join(v1_checkpoint, config_filename)
        )
        if config.hidden_act == "gelu":
            activation = activations.gelu
        else:
            activation = tf.keras.activations.get(config.hidden_act)

        layer = cls(
            vocab_size=config.vocab_size,
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
            )
        )
        bert_network = create_albert_model(config)
        with tempfile.TemporaryDirectory() as temp_dir:
            temporary_checkpoint = os.path.join(temp_dir, "ckpt")
            convert_albert_checkpoint(
                checkpoint_from_path=v1_checkpoint,
                checkpoint_to_path=temporary_checkpoint,
                num_heads=config.num_attention_heads
            )
            bert_network.load_weights(
                temporary_checkpoint
            ).assert_existing_objects_matched()

            with open(os.path.join(v1_checkpoint, "vocab.txt")) as f:
                vocab = f.read().split("\n")
                bert_preprocessing_layer = BertPreprocessingLayer(vocab)

            layer(bert_preprocessing_layer(tf.constant(["是的", "不是的"])))
            layer.embedding_layer.set_weights(
                bert_network.get_layer("word_embeddings").get_weights()
            )
            layer.position_embedding_layer.set_weights(
                bert_network.get_layer("position_embedding").get_weights()
            )
            layer.type_embedding_layer.set_weights(
                bert_network.get_layer("type_embeddings").get_weights()
            )
            layer.normalize_layer.set_weights(
                bert_network.get_layer("embeddings/layer_norm").get_weights()
            )
            layer.shared_layer.set_weights(
                bert_network.get_layer("transformer").get_weights()
            )
            layer.cls_output_layer.set_weights(
                bert_network.get_layer("pooler_transform").get_weights()
            )
            return layer
