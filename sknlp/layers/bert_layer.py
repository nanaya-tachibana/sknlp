from __future__ import annotations
from typing import Sequence, Any, Optional, Callable, Union

import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from official.nlp.keras_nlp import layers
import tensorflow_text as tftext

from sknlp.activations import gelu

from .bert_tokenization import BertTokenizationLayer


@tf.keras.utils.register_keras_serializable(package="sknlp")
class BertCharPreprocessingLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab: Sequence[str],
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        name: str = "bert_tokenize",
        **kwargs,
    ) -> None:
        vocab: list = list(vocab)
        self.cls_token = cls_token
        self.sep_token = sep_token
        if cls_token not in vocab:
            vocab.append(cls_token)
        if sep_token not in vocab:
            vocab.append(sep_token)
        if "[UNK]" in vocab:
            idx = vocab.index("[UNK]")
            vocab[idx] = "unk"
        self.vocab = vocab
        super().__init__(name=name, **kwargs)

    def build(self, input_shape: tf.TensorShape) -> None:
        self.tokenizer = TextVectorization(
            max_tokens=len(self.vocab),
            standardize=None,
            split=BertTokenizationLayer(
                cls_token=self.cls_token, sep_token=self.sep_token
            ),
        )
        self.tokenizer.set_vocabulary(self.vocab[2:])
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> list[tf.Tensor]:
        token_ids = self.tokenizer(inputs)
        return token_ids, tf.zeros_like(token_ids, dtype=token_ids.dtype)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "vocab": self.vocab,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
        }


@tf.keras.utils.register_keras_serializable(package="sknlp")
class BertPreprocessingLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab: Sequence[str],
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        name: str = "bert_tokenize",
        **kwargs,
    ) -> None:
        vocab: list = list(vocab)
        self.cls_token = cls_token
        self.sep_token = sep_token
        if cls_token not in vocab:
            vocab.append(cls_token)
        if sep_token not in vocab:
            vocab.append(sep_token)
        self.vocab = vocab
        for i, token in enumerate(vocab):
            if token == cls_token:
                self.cls_id = i
            if token == sep_token:
                self.sep_id = i
        super().__init__(name=name, **kwargs)

    def build(self, input_shape: tf.TensorShape) -> None:
        self.tokenizer = tftext.BertTokenizer(
            tf.lookup.StaticVocabularyTable(
                tf.lookup.KeyValueTensorInitializer(
                    self.vocab,
                    list(range(len(self.vocab))),
                    key_dtype=tf.string,
                    value_dtype=tf.int64,
                ),
                1,
            )
        )
        super().build(input_shape)

    def _create_special_tensor(self, token_id: int, nrows: tf.Tensor) -> tf.Tensor:
        return tf.expand_dims(
            tf.tile(tf.constant([token_id], dtype=tf.int64), [nrows]), -1
        )

    def _create_cls_tensor(self, nrows: tf.Tensor) -> tf.Tensor:
        return self._create_special_tensor(self.cls_id, nrows)

    def _create_sep_tensor(self, nrows: tf.Tensor) -> tf.Tensor:
        return self._create_special_tensor(self.sep_id, nrows)

    def call(self, inputs: Union[tf.Tensor, list[tf.Tensor]]) -> list[tf.Tensor]:
        inputs = tf.nest.flatten(inputs)
        batch_size = tf.shape(inputs[0])[0]
        cls_ids = self._create_cls_tensor(batch_size)
        sep_ids = self._create_sep_tensor(batch_size)
        if len(inputs) == 1:
            token_ids = self.tokenizer.tokenize(inputs[0])
            flatten_ids = token_ids.merge_dims(-2, -1)
            ids = tf.concat([cls_ids, flatten_ids, sep_ids], axis=1)
            ids_tensor = ids.to_tensor(0)
            return ids_tensor, tf.zeros_like(ids_tensor, dtype=ids_tensor.dtype)
        elif len(inputs) == 2:
            query, context = inputs
            query_token_ids = self.tokenizer.tokenize(query)
            context_token_ids = self.tokenizer.tokenize(context)
            query_flatten_ids = query_token_ids.merge_dims(-2, -1)
            context_flatten_ids = context_token_ids.merge_dims(-2, -1)
            query_ids = tf.concat([cls_ids, query_flatten_ids, sep_ids], axis=1)
            context_ids = tf.concat([context_flatten_ids, sep_ids], axis=1)
            type_ids = tf.concat(
                [
                    tf.zeros_like(query_ids, dtype=tf.int64),
                    tf.ones_like(context_ids, dtype=tf.int64),
                ],
                axis=1,
            )
            return (
                tf.concat([query_ids, context_ids], axis=1, name="token_ids").to_tensor(
                    0
                ),
                type_ids.to_tensor(0),
            )
        else:
            raise ValueError(
                f"The length of inputs must be 1 or 2, bug get {len(inputs)}."
            )

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "vocab": self.vocab,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
        }


@tf.keras.utils.register_keras_serializable(package="sknlp")
class BertLayer(tf.keras.layers.Layer):
    """
    copy from https://github.com/tensorflow/models/blob/v2.2.0/official/nlp/modeling/networks/transformer_encoder.py.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        embedding_size: Optional[int] = 768,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        sequence_length: int = None,
        max_sequence_length: int = 512,
        type_vocab_size: int = 16,
        intermediate_size: int = 3072,
        activation: Callable = gelu,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        initializer: tf.keras.initializers.Initializer = TruncatedNormal(stddev=0.02),
        share_layer: bool = False,
        cls_pooling: bool = True,
        name: str = "bert_layer",
        **kwargs,
    ) -> None:
        if embedding_size is None:
            embedding_size = hidden_size

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
        self.share_layer = share_layer
        self.cls_pooling = cls_pooling
        super().__init__(name=name, **kwargs)

    def build(self, input_shape: tf.TensorShape) -> None:
        input_shape = input_shape[0]
        self.embedding_layer = layers.OnDeviceEmbedding(
            self.vocab_size,
            self.embedding_size,
            initializer=self.initializer,
            name="word_embeddings",
        )
        self.position_embedding_layer = layers.PositionEmbedding(
            self.max_sequence_length,
            initializer=self.initializer,
            name="position_embedding",
        )
        self.type_embedding_layer = layers.OnDeviceEmbedding(
            self.type_vocab_size,
            self.embedding_size,
            initializer=self.initializer,
            use_one_hot=True,
            name="type_embeddings",
        )
        self.embedding_normalize_layer = tf.keras.layers.LayerNormalization(
            name="embeddings/layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32
        )
        self.embedding_dropout_layer = tf.keras.layers.Dropout(
            rate=self.dropout_rate, name="embeddings/dropout"
        )
        if self.embedding_size != self.hidden_size:
            self.embedding_projection = tf.keras.layers.experimental.EinsumDense(
                "...x,xy->...y",
                output_shape=self.hidden_size,
                bias_axes="y",
                kernel_initializer=self.initializer,
                name="embeddings/transform",
            )
        else:
            self.embedding_projection = None

        if self.share_layer:
            self.shared_layer = layers.TransformerEncoderBlock(
                self.num_attention_heads,
                self.intermediate_size,
                self.activation,
                output_dropout=self.dropout_rate,
                attention_dropout=self.attention_dropout_rate,
                kernel_initializer=self.initializer,
                name="transformer",
            )
        else:
            self.transformer_layers = []
            for i in range(self.num_layers):
                layer = layers.TransformerEncoderBlock(
                    self.num_attention_heads,
                    self.intermediate_size,
                    self.activation,
                    output_dropout=self.dropout_rate,
                    attention_dropout=self.attention_dropout_rate,
                    kernel_initializer=self.initializer,
                    name="transformer/layer_%d" % i,
                )
                self.transformer_layers.append(layer)

        if self.cls_pooling:
            self.cls_output_layer = tf.keras.layers.Dense(
                units=self.hidden_size,
                activation="tanh",
                kernel_initializer=self.initializer,
                name="cls_pooler",
            )
        self.lm_dense = tf.keras.layers.Dense(
            self.embedding_size,
            activation=self.activation,
            kernel_initializer=self.initializer,
            name="lm/transform/dense",
        )
        self.lm_normalize_layer = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12, name="lm/transform/layer_norm"
        )
        self.lm_bias = self.add_weight(
            "lm/output_bias",
            shape=(self.vocab_size,),
            initializer="zeros",
        )
        self.relationship_dense = tf.keras.layers.Dense(
            2, kernel_initializer=self.initializer, name="relationship"
        )
        super().build(input_shape)

    def compute_output_shape(
        self, input_shape: list[tf.TensorShape]
    ) -> list[tf.TensorShape]:
        return super().compute_output_shape(input_shape)

    def call(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        token_ids, type_ids, attention_mask, logits_mask = inputs
        word_embeddings = self.embedding_layer(token_ids)
        position_embeddings = self.position_embedding_layer(word_embeddings)
        type_embeddings = self.type_embedding_layer(type_ids)
        embeddings = tf.keras.layers.Add()(
            [word_embeddings, position_embeddings, type_embeddings]
        )
        embeddings = self.embedding_normalize_layer(embeddings)
        embeddings = self.embedding_dropout_layer(embeddings)
        if self.embedding_projection is not None:
            embeddings = self.embedding_projection(embeddings)
        data = embeddings
        encoder_outputs = []
        if self.share_layer:
            for _ in range(self.num_layers):
                data = self.shared_layer([data, attention_mask])
                encoder_outputs.append(data)
        else:
            for layer in self.transformer_layers:
                data = layer([data, attention_mask])
                encoder_outputs.append(data)

        first_token_tensor = tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x[:, 0:1, :], axis=1)
        )(encoder_outputs[-1])
        cls_output = first_token_tensor
        if self.cls_pooling:
            cls_output = self.cls_output_layer(cls_output)
        # (batch_size, mask_seq_len, dim)
        token_encodings = tf.ragged.boolean_mask(
            encoder_outputs[-1], tf.cast(logits_mask, tf.bool)
        ).to_tensor()
        # (batch_size, mask_seq_len, embedding_size)
        lm_data = self.lm_dense(token_encodings)
        lm_data = self.lm_normalize_layer(lm_data)
        # (batch_size, mask_seq_len, vocab_size)
        lm_logits = tf.nn.bias_add(
            tf.matmul(lm_data, self.embedding_layer.embeddings, transpose_b=True),
            self.lm_bias,
        )
        rel_logits = self.relationship_dense(cls_output)
        return cls_output, encoder_outputs, lm_logits, rel_logits

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
            "share_layer": self.share_layer,
            "cls_pooling": self.cls_pooling,
        }