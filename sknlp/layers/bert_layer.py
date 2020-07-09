import os
import tempfile
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from official.modeling import activations
from official.nlp.modeling import layers
from official.nlp.modeling import networks
from official.nlp.bert import configs
import tensorflow.compat.v1 as tfv1

from .bert_tokenization import BertTokenizationLayer


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class BertPreprocessingLayer(tf.keras.layers.Layer):

    def __init__(self, vocab, sequence_length=100, **kwargs):
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

    def call(self, inputs):
        return self.text2vec(inputs)

    def get_config(self):
        return {
            **super().get_config(),
            "vocab": self.vocab,
            "sequence_length": self.sequence_length
        }

BERT_NAME_REPLACEMENTS = (
    ("bert/", ""),
    ("encoder", "transformer"),
    ("embeddings/word_embeddings", "word_embeddings/embeddings"),
    ("embeddings/token_type_embeddings", "type_embeddings/embeddings"),
    ("embeddings/position_embeddings", "position_embedding/embeddings"),
    ("embeddings/LayerNorm", "embeddings/layer_norm"),
    ("attention/self", "self_attention"),
    ("attention/output/dense", "self_attention_output"),
    ("attention/output/LayerNorm", "self_attention_layer_norm"),
    ("intermediate/dense", "intermediate"),
    ("output/dense", "output"),
    ("output/LayerNorm", "output_layer_norm"),
    ("pooler/dense", "pooler_transform"),
    ("cls/predictions/output_bias", "cls/predictions/output_bias/bias"),
    ("cls/seq_relationship/output_bias", "predictions/transform/logits/bias"),
    ("cls/seq_relationship/output_weights", "predictions/transform/logits/kernel"),
)
BERT_PERMUTATIONS = (("cls/seq_relationship/output_weights", (1, 0)),)


def _bert_name_replacement(var_name, name_replacements):
  """Gets the variable name replacement."""
  for src_pattern, tgt_pattern in name_replacements:
    if src_pattern in var_name:
      old_var_name = var_name
      var_name = var_name.replace(src_pattern, tgt_pattern)
      logger.info("Converted: %s --> %s", old_var_name, var_name)
  return var_name


def _has_exclude_patterns(name, exclude_patterns):
  """Checks if a string contains substrings that match patterns to exclude."""
  for p in exclude_patterns:
    if p in name:
      return True
  return False


def _get_permutation(name, permutations):
  """Checks whether a variable requires transposition by pattern matching."""
  for src_pattern, permutation in permutations:
    if src_pattern in name:
      logger.info("Permuted: %s --> %s", name, permutation)
      return permutation

  return None


def _get_new_shape(name, shape, num_heads):
  """Checks whether a variable requires reshape by pattern matching."""
  if "self_attention_output/kernel" in name:
    return (num_heads, shape[0] // num_heads, shape[1])
  if "self_attention_output/bias" in name:
    return shape

  patterns = [
      "self_attention/query", "self_attention/value", "self_attention/key"
  ]
  for pattern in patterns:
    if pattern in name:
      if "kernel" in name:
        return tuple([shape[0], num_heads, shape[1] // num_heads])
      if "bias" in name:
        return tuple([num_heads, shape[0] // num_heads])
  return None


def convert_checkpoint(checkpoint_from_path,
                       checkpoint_to_path,
                       num_heads,
                       name_replacements,
                       permutations,
                       exclude_patterns=None):
  """Migrates the names of variables within a checkpoint.
  Args:
    checkpoint_from_path: Path to source checkpoint to be read in.
    checkpoint_to_path: Path to checkpoint to be written out.
    num_heads: The number of heads of the model.
    name_replacements: A list of tuples of the form (match_str, replace_str)
      describing variable names to adjust.
    permutations: A list of tuples of the form (match_str, permutation)
      describing permutations to apply to given variables. Note that match_str
      should match the original variable name, not the replaced one.
    exclude_patterns: A list of string patterns to exclude variables from
      checkpoint conversion.
  Returns:
    A dictionary that maps the new variable names to the Variable objects.
    A dictionary that maps the old variable names to the new variable names.
  """
  with tfv1.Graph().as_default():
    logger.info("Reading checkpoint_from_path %s", checkpoint_from_path)
    reader = tfv1.train.load_checkpoint(checkpoint_from_path)
    name_shape_map = reader.get_variable_to_shape_map()
    new_variable_map = {}
    conversion_map = {}
    for var_name in name_shape_map:
      if exclude_patterns and _has_exclude_patterns(var_name, exclude_patterns):
        continue
      # Get the original tensor data.
      tensor = reader.get_tensor(var_name)

      # Look up the new variable name, if any.
      new_var_name = _bert_name_replacement(var_name, name_replacements)

      # See if we need to reshape the underlying tensor.
      new_shape = None
      if num_heads > 0:
        new_shape = _get_new_shape(new_var_name, tensor.shape, num_heads)
      if new_shape:
        logger.info("Veriable %s has a shape change from %s to %s",
                    var_name, tensor.shape, new_shape)
        tensor = np.reshape(tensor, new_shape)

      # See if we need to permute the underlying tensor.
      permutation = _get_permutation(var_name, permutations)
      if permutation:
        tensor = np.transpose(tensor, permutation)

      # Create a new variable with the possibly-reshaped or transposed tensor.
      var = tfv1.Variable(tensor, name=var_name)

      # Save the variable into the new variable map.
      new_variable_map[new_var_name] = var

      # Keep a list of converter variables for sanity checking.
      if new_var_name != var_name:
        conversion_map[var_name] = new_var_name

    saver = tfv1.train.Saver(new_variable_map)

    with tfv1.Session() as sess:
      sess.run(tfv1.global_variables_initializer())
      logger.info("Writing checkpoint_to_path %s", checkpoint_to_path)
      saver.save(sess, checkpoint_to_path, write_meta_graph=True)

  logger.info("Summary:")
  logger.info("  Converted %d variable name(s).", len(new_variable_map))
  logger.info("  Converted: %s", str(conversion_map))



def create_bert_model(cfg):
    """Creates a BERT keras core model from BERT configuration.
    Args:
    cfg: A `BertConfig` to create the core model.
    Returns:
    A TransformerEncoder netowork.
    """
    bert_encoder = networks.TransformerEncoder(
      vocab_size=cfg.vocab_size,
      hidden_size=cfg.hidden_size,
      num_layers=cfg.num_hidden_layers,
      num_attention_heads=cfg.num_attention_heads,
      intermediate_size=cfg.intermediate_size,
      activation=activations.gelu,
      dropout_rate=cfg.hidden_dropout_prob,
      attention_dropout_rate=cfg.attention_probs_dropout_prob,
      sequence_length=80,
      max_sequence_length=cfg.max_position_embeddings,
      type_vocab_size=cfg.type_vocab_size,
      initializer=tf.keras.initializers.TruncatedNormal(stddev=cfg.initializer_range)
    )
    return bert_encoder


class BertLayer(tf.keras.layers.Layer):
    """
    copy from https://github.com/tensorflow/models/blob/v2.2.0/official/nlp/modeling/networks/transformer_encoder.py.
    """

    def __init__(
        self,
        vocab_size,
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
        initializer=None, # tf.keras.initializers.TruncatedNormal(stddev=0.02),
        return_all_encoder_outputs=False,
        name="bert_layer",
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
            "activation": tf.keras.activations.serialize(self.activation),
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "initializer": self.initializer,
            "return_all_encoder_outputs": self.return_all_encoder_outputs
        }

    @classmethod
    def from_tfv1_checkpoint(cls, v1_checkpoint, sequence_length=100):
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
            convert_checkpoint(
                checkpoint_from_path=v1_checkpoint,
                checkpoint_to_path=temporary_checkpoint,
                num_heads=bert_config.num_attention_heads,
                name_replacements=BERT_NAME_REPLACEMENTS,
                permutations=BERT_PERMUTATIONS,
                exclude_patterns=["adam", "Adam"]
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
