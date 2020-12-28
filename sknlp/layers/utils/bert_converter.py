from typing import List, Optional, Sequence
import logging
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


BERT_NAME_REPLACEMENTS = [
    ("embeddings_project", "embedding_projection"),
    ("encoder", "transformer"),
    ("embeddings/word_embeddings", "word_embeddings/embeddings"),
    ("embeddings/token_type_embeddings", "type_embeddings/embeddings"),
    ("embeddings/position_embeddings", "position_embedding/embeddings"),
    ("embeddings/LayerNorm", "embeddings/layer_norm"),
    ("attention/self", "self_attention"),
    ("attention/output/dense", "self_attention/attention_output"),
    ("attention/output/LayerNorm", "self_attention_layer_norm"),
    ("intermediate/dense", "intermediate"),
    ("output/dense", "output"),
    ("output/LayerNorm", "output_layer_norm"),
    ("pooler/dense", "pooler_transform"),
    ("cls/predictions/output_bias", "cls/predictions/output_bias/bias"),
    ("cls/seq_relationship/output_bias", "predictions/transform/logits/bias"),
    ("cls/seq_relationship/output_weights", "predictions/transform/logits/kernel"),
]
BERT_PERMUTATIONS = [
    ("cls/seq_relationship/output_weights", (1, 0)),
]


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
    if "self_attention/attention_output/kernel" in name:
        return (num_heads, shape[0] // num_heads, shape[1])
    if "self_attention_output/bias" in name:
        return shape

    patterns = ["self_attention/query", "self_attention/value", "self_attention/key"]
    for pattern in patterns:
        if pattern in name:
            if "kernel" in name:
                return tuple([shape[0], num_heads, shape[1] // num_heads])
            if "bias" in name:
                return tuple([num_heads, shape[0] // num_heads])
    return None


def convert_checkpoint(
    checkpoint_from_path: str,
    checkpoint_to_path: str,
    num_heads: int,
    original_root_name: str = "bert",
    converted_root_name: str = "bert",
    name_replacements: Optional[Sequence] = None,
    permutations: Optional[Sequence] = None,
    exclude_patterns: Optional[Sequence] = None,
) -> None:
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
    name_replacements = list(name_replacements) or list()
    if converted_root_name != "":
        name_replacements.insert(0, (original_root_name, converted_root_name))
    else:
        name_replacements.insert(0, (original_root_name + "/", ""))
    permutations = permutations or list()
    exclude_patterns = exclude_patterns or ["adam", "Adam"]
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
                logger.info(
                    "Veriable %s has a shape change from %s to %s",
                    var_name,
                    tensor.shape,
                    new_shape,
                )
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


convert_bert_checkpoint = partial(
    convert_checkpoint,
    name_replacements=BERT_NAME_REPLACEMENTS,
    permutations=BERT_PERMUTATIONS,
)
convert_electra_checkpoint = partial(
    convert_bert_checkpoint, original_root_name="electra"
)
