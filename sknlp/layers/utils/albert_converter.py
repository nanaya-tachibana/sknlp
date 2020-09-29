import logging
from functools import partial

import tensorflow as tf
from official.modeling import activations
from official.nlp.modeling import networks

from .bert_converter import convert_checkpoint, BERT_PERMUTATIONS


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


ALBERT_NAME_REPLACEMENTS = (
    ("bert/encoder/", ""),
    ("bert/", ""),
    ("embeddings/word_embeddings", "word_embeddings/embeddings"),
    ("embeddings/position_embeddings", "position_embedding/embeddings"),
    ("embeddings/token_type_embeddings", "type_embeddings/embeddings"),
    ("embeddings/LayerNorm", "embeddings/layer_norm"),
    ("embedding_hidden_mapping_in", "embedding_projection"),
    ("group_0/inner_group_0/", ""),
    ("attention_1/self", "self_attention"),
    ("attention_1/output/dense", "self_attention/attention_output"),
    ("LayerNorm/", "self_attention_layer_norm/"),
    ("ffn_1/intermediate/dense", "intermediate"),
    ("ffn_1/intermediate/output/dense", "output"),
    ("LayerNorm_1/", "output_layer_norm/"),
    ("pooler/dense", "pooler_transform"),
    ("cls/predictions/output_bias", "cls/predictions/output_bias/bias"),
    ("cls/seq_relationship/output_bias", "predictions/transform/logits/bias"),
    ("cls/seq_relationship/output_weights",
     "predictions/transform/logits/kernel"),
)



convert_albert_checkpoint = partial(convert_checkpoint,
                                    name_replacements=ALBERT_NAME_REPLACEMENTS,
                                    permutations=BERT_PERMUTATIONS)




def create_albert_model(cfg):
  """Creates a BERT keras core model from BERT configuration.
  Args:
    cfg: A `BertConfig` to create the core model.
  Returns:
    A keras model.
  """
  albert_encoder = networks.AlbertTransformerEncoder(
      vocab_size=cfg.vocab_size,
      hidden_size=cfg.hidden_size,
      embedding_width=cfg.embedding_size,
      num_layers=cfg.num_hidden_layers,
      num_attention_heads=cfg.num_attention_heads,
      intermediate_size=cfg.intermediate_size,
      activation=activations.gelu,
      dropout_rate=cfg.hidden_dropout_prob,
      attention_dropout_rate=cfg.attention_probs_dropout_prob,
      max_sequence_length=cfg.max_position_embeddings,
      type_vocab_size=cfg.type_vocab_size,
      initializer=tf.keras.initializers.TruncatedNormal(stddev=cfg.initializer_range))
  return albert_encoder
