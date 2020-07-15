from typing import List, Dict, Any
import os
import tempfile

import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from official.modeling import activations
from official.nlp.modeling import layers
from official.nlp.bert import configs

from .bert_tokenization import BertTokenizationLayer
from .utils import convert_bert_checkpoint, create_bert_model


class BertPreprocessingLayer(tf.keras.layers.Layer):

    def __init__(self, vocab: List[str], sequence_length: int = None, **kwargs) -> None:
        self.vocab = vocab
        self.sequence_length = sequence_length
        self.text2vec = TextVectorization(
            max_tokens=len(vocab) + 3,
            output_sequence_length=sequence_length,
            standardize=None,
            split=BertTokenizationLayer()
        )
        self.text2vec.set_vocabulary(vocab)
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.text2vec(inputs)

    def get_config(self) -> Dict[str, Any]:
        return {
            **super().get_config(),
            "vocab": self.vocab,
            "sequence_length": self.sequence_length
        }


class BertLayer(tf.keras.layers.Layer):
    """
    copy from https://github.com/tensorflow/models/blob/v2.2.0/official/nlp/modeling/networks/transformer_encoder.py.
    """

    def __init__(
        self,
        vocab_size: int,
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
    ):
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
            vocab_size=vocab_size,
            embedding_width=hidden_size,
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
            vocab_size=type_vocab_size,
            embedding_width=hidden_size,
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

    def call(self, inputs):
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
    def from_tfv1_checkpoint(cls, v1_checkpoint, sequence_length=None):
        bert_config = configs.BertConfig.from_json_file(
            os.path.join(v1_checkpoint, "bert_config.json")
        )
        layer = cls(
            vocab_size=bert_config.vocab_size,
            hidden_size=bert_config.hidden_size,
            num_layers=bert_config.num_hidden_layers,
            num_attention_heads=bert_config.num_attention_heads,
            intermediate_size=bert_config.intermediate_size,
            activation=activations.gelu,
            dropout_rate=bert_config.hidden_dropout_prob,
            attention_dropout_rate=bert_config.attention_probs_dropout_prob,
            sequence_length=sequence_length,
            max_sequence_length=bert_config.max_position_embeddings,
            type_vocab_size=bert_config.type_vocab_size,
            initializer=tf.keras.initializers.TruncatedNormal(
                stddev=bert_config.initializer_range
            )
        )
        bert_network = create_bert_model(bert_config)
        with tempfile.TemporaryDirectory() as temp_dir:
            temporary_checkpoint = os.path.join(temp_dir, "ckpt")
            convert_bert_checkpoint(
                checkpoint_from_path=v1_checkpoint,
                checkpoint_to_path=temporary_checkpoint,
                num_heads=bert_config.num_attention_heads
            )
            bert_network.load_weights(
                temporary_checkpoint
            ).assert_existing_objects_matched()

            with open(os.path.join(v1_checkpoint, "vocab.txt")) as f:
                vocab = f.read().split("\n")
                bert_preprocessing_layer = BertPreprocessingLayer(
                    vocab[2:], sequence_length
                )

            layer(bert_preprocessing_layer(tf.constant([["是的"], ["不是的"]])))
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
            for i in range(bert_config.num_hidden_layers):
                layer.transformer_layers[i].set_weights(
                    bert_network.get_layer("transformer/layer_%d" % i).get_weights()
                )
            layer.cls_output_layer.set_weights(
                bert_network.get_layer("pooler_transform").get_weights()
            )
            return layer
