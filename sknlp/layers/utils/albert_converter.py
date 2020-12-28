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
    ("encoder/", ""),
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
    ("cls/seq_relationship/output_weights", "predictions/transform/logits/kernel"),
)


convert_albert_checkpoint = partial(
    convert_checkpoint,
    name_replacements=ALBERT_NAME_REPLACEMENTS,
    permutations=BERT_PERMUTATIONS,
)
