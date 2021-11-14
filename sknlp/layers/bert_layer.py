# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import annotations
from typing import Sequence, Any, Optional, Callable

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LayerNormalization, Dropout
from tensorflow.keras.initializers import TruncatedNormal

import tensorflow_text as tftext

from sknlp.activations import gelu

from .transformer_encoder_block import TransformerEncoderBlock


@tf.keras.utils.register_keras_serializable(package="sknlp")
class BertPreprocessingLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab: Sequence[str],
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        max_length: int = 510,
        return_offset: bool = False,
        name: str = "bert_tokenize",
        **kwargs,
    ) -> None:
        vocab: list = list(vocab)
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.max_length = max_length
        if cls_token not in vocab:
            vocab.append(cls_token)
        if sep_token not in vocab:
            vocab.append(sep_token)

        self.max_chars_per_token = 1
        for i, token in enumerate(vocab):
            if token == cls_token:
                self.cls_id = i
            if token == sep_token:
                self.sep_id = i
            self.max_chars_per_token = max(self.max_chars_per_token, len(token))
        self.vocab = vocab
        self.return_offset = return_offset
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
            ),
            max_chars_per_token=self.max_chars_per_token,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor | list[tf.Tensor]) -> list[tf.Tensor]:
        inputs = tf.nest.flatten(inputs)
        trimmer = tftext.WaterfallTrimmer(self.max_length)
        if len(inputs) == 1:
            if self.return_offset:
                (token_ids, starts, ends) = self.tokenizer.tokenize_with_offsets(
                    inputs[0]
                )
                starts = starts.merge_dims(-2, -1)
                ends = ends.merge_dims(-2, -1)
            else:
                token_ids = self.tokenizer.tokenize(inputs[0])
            flatten_ids = token_ids.merge_dims(-2, -1)
            ids, type_ids = tftext.combine_segments(
                trimmer.trim([flatten_ids]),
                start_of_sequence_id=self.cls_id,
                end_of_segment_id=self.sep_id,
            )
            tensors: list[tf.Tensor] = [ids.to_tensor(), type_ids.to_tensor()]
            if self.return_offset:
                tensors.append(starts.to_tensor())
                tensors.append(ends.to_tensor())
            return tensors
        elif len(inputs) == 2:
            query, context = inputs
            query_token_ids = self.tokenizer.tokenize(query)
            if self.return_offset:
                # TODO: 句对输入仅输出第二句的start和end索引是否合理,
                #       需要根据具体任务设计判断, 如阅读理解
                (
                    context_token_ids,
                    starts,
                    ends,
                ) = self.tokenizer.tokenize_with_offsets(context)
                starts = starts.merge_dims(-2, -1)
                ends = ends.merge_dims(-2, -1)
            else:
                context_token_ids = self.tokenizer.tokenize(context)
            query_flatten_ids = query_token_ids.merge_dims(-2, -1)
            context_flatten_ids = context_token_ids.merge_dims(-2, -1)
            token_ids, type_ids = tftext.combine_segments(
                trimmer.trim([query_flatten_ids, context_flatten_ids]),
                start_of_sequence_id=self.cls_id,
                end_of_segment_id=self.sep_id,
            )

            tensors: list[tf.Tensor] = [token_ids.to_tensor(), type_ids.to_tensor()]
            if self.return_offset:
                tensors.append(starts.to_tensor())
                tensors.append(ends.to_tensor())
            return tensors
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
            "return_offset": self.return_offset,
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
        enable_recompute_grad: bool = False,
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
        self.enable_recompute_grad = enable_recompute_grad
        super().__init__(name=name, **kwargs)

    def build(self, input_shape: tf.TensorShape) -> None:
        input_shape = input_shape[0]
        self.embedding_layer = Embedding(
            self.vocab_size,
            self.embedding_size,
            embeddings_initializer=self.initializer,
            name="word_embeddings",
        )
        self.position_embedding_layer = Embedding(
            self.max_sequence_length,
            self.embedding_size,
            embeddings_initializer=self.initializer,
            name="position_embedding",
        )
        self.type_embedding_layer = Embedding(
            self.type_vocab_size,
            self.embedding_size,
            embeddings_initializer=self.initializer,
            name="type_embeddings",
        )
        self.embedding_normalize_layer = LayerNormalization(
            name="embeddings/layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32
        )
        self.embedding_dropout_layer = Dropout(
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
            self.shared_layer = TransformerEncoderBlock(
                self.num_attention_heads,
                self.intermediate_size,
                self.activation,
                output_dropout=self.dropout_rate,
                attention_dropout=self.attention_dropout_rate,
                kernel_initializer=self.initializer,
                enable_recompute_grad=self.enable_recompute_grad,
                name="transformer",
            )
        else:
            self.transformer_layers = []
            for i in range(self.num_layers):
                layer = TransformerEncoderBlock(
                    self.num_attention_heads,
                    self.intermediate_size,
                    self.activation,
                    output_dropout=self.dropout_rate,
                    attention_dropout=self.attention_dropout_rate,
                    kernel_initializer=self.initializer,
                    enable_recompute_grad=self.enable_recompute_grad,
                    name="transformer/layer_%d" % i,
                )
                self.transformer_layers.append(layer)

        if self.cls_pooling:
            self.cls_output_layer = Dense(
                units=self.hidden_size,
                activation="tanh",
                kernel_initializer=self.initializer,
                name="cls_pooler",
            )
        self.lm_dense = Dense(
            self.embedding_size,
            activation=self.activation,
            kernel_initializer=self.initializer,
            name="lm/transform/dense",
        )
        self.lm_normalize_layer = LayerNormalization(
            axis=-1, epsilon=1e-12, name="lm/transform/layer_norm"
        )
        self.lm_bias = self.add_weight(
            "lm/output_bias",
            shape=(self.vocab_size,),
            initializer="zeros",
        )
        self.relationship_dense = Dense(
            2, kernel_initializer=self.initializer, name="relationship"
        )
        super().build(input_shape)

    def compute_output_shape(
        self, input_shape: list[tf.TensorShape]
    ) -> list[tf.TensorShape]:
        return super().compute_output_shape(input_shape)

    def call(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        token_ids, type_ids, attention_mask, logits_mask = inputs
        # (batch_size, seq_len, embedding_size)
        word_embeddings = self.embedding_layer(token_ids)
        # (1, seq_len)
        position_ids = tf.range(tf.shape(token_ids)[-1])[None, ...]
        # (1, seq_len, embedding_size)
        position_embeddings = self.position_embedding_layer(position_ids)
        # (batch_size, seq_len, embedding_size)
        type_embeddings = self.type_embedding_layer(type_ids)
        embeddings = word_embeddings + position_embeddings + type_embeddings
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

        first_token_tensor = encoder_outputs[-1][:, 0, :]
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
            "enable_recompute_grad": self.enable_recompute_grad,
        }