from __future__ import annotations

import re
import warnings

import numpy as np
import tensorflow as tf

from sknlp.utils.logging import logger


class BertCheckpointConverter:
    def __init__(
        self, num_layers: int, num_heads: int, converted_root_name: str = ""
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.converted_root_name = converted_root_name
        self.variable_name_mapping: dict[str, str] = dict()
        for mapping in (
            self.embeddings,
            self.embedding_transform,
            self.transformer,
            self.cls_pooler,
        ):
            self.variable_name_mapping.update(
                {
                    self.prefix(k): self.prefix(v, original=False)
                    for k, v in mapping.items()
                }
            )
        for mapping in (self.lm, self.relationship):
            self.variable_name_mapping.update(
                {k: self.prefix(v, original=False) for k, v in mapping.items()}
            )

    @property
    def original_root_name(self) -> str:
        return "bert"

    def prefix(self, name: str, original: bool = True) -> str:
        if original:
            root_name = self.original_root_name
        else:
            root_name = self.converted_root_name
        if root_name:
            root_name += "/"
        return root_name + name

    @property
    def embeddings(self) -> dict[str, str]:
        return {
            "embeddings/LayerNorm/beta": "embeddings/layer_norm/beta",
            "embeddings/LayerNorm/gamma": "embeddings/layer_norm/gamma",
            "embeddings/position_embeddings": "position_embedding/embeddings",
            "embeddings/token_type_embeddings": "type_embeddings/embeddings",
            "embeddings/word_embeddings": "word_embeddings/embeddings",
        }

    @property
    def embedding_transform(self) -> dict[str, str]:
        return {
            "embeddings_project/kernel": "embeddings/transform/kernel",
            "embeddings_project/bias": "embeddings/transform/bias",
        }

    @property
    def transformer(self) -> dict[str, str]:
        mapping = dict()
        for i in range(self.num_layers):
            mapping.update(
                {
                    f"encoder/layer_{i}/attention/self/key/bias": f"transformer/layer_{i}/self_attention/key/bias",
                    f"encoder/layer_{i}/attention/self/key/kernel": f"transformer/layer_{i}/self_attention/key/kernel",
                    f"encoder/layer_{i}/attention/self/query/bias": f"transformer/layer_{i}/self_attention/query/bias",
                    f"encoder/layer_{i}/attention/self/query/kernel": f"transformer/layer_{i}/self_attention/query/kernel",
                    f"encoder/layer_{i}/attention/self/value/bias": f"transformer/layer_{i}/self_attention/value/bias",
                    f"encoder/layer_{i}/attention/self/value/kernel": f"transformer/layer_{i}/self_attention/value/kernel",
                    f"encoder/layer_{i}/attention/output/dense/bias": f"transformer/layer_{i}/self_attention/attention_output/bias",
                    f"encoder/layer_{i}/attention/output/dense/kernel": f"transformer/layer_{i}/self_attention/attention_output/kernel",
                    f"encoder/layer_{i}/attention/output/LayerNorm/beta": f"transformer/layer_{i}/self_attention_layer_norm/beta",
                    f"encoder/layer_{i}/attention/output/LayerNorm/gamma": f"transformer/layer_{i}/self_attention_layer_norm/gamma",
                    f"encoder/layer_{i}/intermediate/dense/bias": f"transformer/layer_{i}/intermediate/bias",
                    f"encoder/layer_{i}/intermediate/dense/kernel": f"transformer/layer_{i}/intermediate/kernel",
                    f"encoder/layer_{i}/output/dense/bias": f"transformer/layer_{i}/output/bias",
                    f"encoder/layer_{i}/output/dense/kernel": f"transformer/layer_{i}/output/kernel",
                    f"encoder/layer_{i}/output/LayerNorm/beta": f"transformer/layer_{i}/output_layer_norm/beta",
                    f"encoder/layer_{i}/output/LayerNorm/gamma": f"transformer/layer_{i}/output_layer_norm/gamma",
                }
            )
        return mapping

    @property
    def cls_pooler(self) -> dict[str, str]:
        return {
            "pooler/dense/bias": "cls_pooler/bias",
            "pooler/dense/kernel": "cls_pooler/kernel",
        }

    @property
    def lm(self) -> dict[str, str]:
        return {
            "cls/predictions/output_bias": "lm/output_bias",
            "cls/predictions/transform/LayerNorm/beta": "lm/transform/layer_norm/beta",
            "cls/predictions/transform/LayerNorm/gamma": "lm/transform/layer_norm/gamma",
            "cls/predictions/transform/dense/bias": "lm/transform/dense/bias",
            "cls/predictions/transform/dense/kernel": "lm/transform/dense/kernel",
        }

    @property
    def relationship(self) -> dict[str, str]:
        return {
            "cls/seq_relationship/output_bias": "relationship/bias",
            "cls/seq_relationship/output_weights": "relationship/kernel",
        }

    @property
    def exclude_keywords(self) -> set[str]:
        return {"adam", "Adam", "global_step"}

    def has_exclude_keywords(self, variable_name: str) -> bool:
        for keyword in self.exclude_keywords:
            if keyword in variable_name:
                return True
        return False

    def reshape_tensor(self, variable_name: str, tensor: np.ndarray) -> np.ndarray:
        r = re.search(
            "self_attention/(query|key|value|attention_output)/(kernel|bias)",
            variable_name,
        )
        if r is None:
            return tensor
        layer, var = r.groups()
        if layer == "attention_output" and var == "bias":
            return tensor

        num_heads = self.num_heads
        shape: tuple = tensor.shape
        if layer == "attention_output" and var == "kernel":
            return tensor.reshape((num_heads, shape[0] // num_heads, shape[1]))
        elif var == "kernel":
            return tensor.reshape((shape[0], num_heads, shape[1] // num_heads))
        else:
            return tensor.reshape((num_heads, shape[0] // num_heads))

    def transpose_tensor(self, variable_name: str, tensor: np.ndarray) -> np.ndarray:
        r = re.search("relationship/kernel", variable_name)
        if r is None:
            return tensor
        return tensor.transpose((1, 0))

    def convert(self, checkpoint_directory: str) -> dict[str, np.ndarray]:
        with tf.Graph().as_default():
            logger.info("Reading checkpoint from: %s", checkpoint_directory)
            name_shape_pairs = tf.train.list_variables(checkpoint_directory)
            new_variables = {}
            for var_name, _ in name_shape_pairs:
                if self.has_exclude_keywords(var_name):
                    continue
                tensor: np.ndarray = tf.train.load_variable(
                    checkpoint_directory, var_name
                )
                new_var_name = self.variable_name_mapping.get(var_name, None)
                if new_var_name is None:
                    warnings.warn(f"Missing convertion rule for variable {var_name}.")
                    continue

                if self.num_heads > 0:
                    tensor = self.reshape_tensor(new_var_name, tensor)
                tensor = self.transpose_tensor(new_var_name, tensor)
                new_variables[new_var_name] = tensor
            return new_variables


class AlbertCheckpointConverter(BertCheckpointConverter):
    @property
    def embedding_transform(self) -> dict[str, str]:
        return {
            "encoder/embedding_hidden_mapping_in/kernel": "embeddings/transform/kernel",
            "encoder/embedding_hidden_mapping_in/bias": "embeddings/transform/bias",
        }

    @property
    def transformer(self) -> dict[str, str]:
        prefix = "encoder/transformer/group_0/inner_group_0"
        return {
            f"{prefix}/attention_1/self/key/bias": "transformer/self_attention/key/bias",
            f"{prefix}/attention_1/self/key/kernel": "transformer/self_attention/key/kernel",
            f"{prefix}/attention_1/self/query/bias": "transformer/self_attention/query/bias",
            f"{prefix}/attention_1/self/query/kernel": "transformer/self_attention/query/kernel",
            f"{prefix}/attention_1/self/value/bias": "transformer/self_attention/value/bias",
            f"{prefix}/attention_1/self/value/kernel": "transformer/self_attention/value/kernel",
            f"{prefix}/attention_1/output/dense/bias": "transformer/self_attention/attention_output/bias",
            f"{prefix}/attention_1/output/dense/kernel": "transformer/self_attention/attention_output/kernel",
            f"{prefix}/LayerNorm/beta": "transformer/self_attention_layer_norm/beta",
            f"{prefix}/LayerNorm/gamma": "transformer/self_attention_layer_norm/gamma",
            f"{prefix}/ffn_1/intermediate/dense/bias": "transformer/intermediate/bias",
            f"{prefix}/ffn_1/intermediate/dense/kernel": "transformer/intermediate/kernel",
            f"{prefix}/ffn_1/intermediate/output/dense/bias": "transformer/output/bias",
            f"{prefix}/ffn_1/intermediate/output/dense/kernel": "transformer/output/kernel",
            f"{prefix}/LayerNorm_1/beta": "transformer/output_layer_norm/beta",
            f"{prefix}/LayerNorm_1/gamma": "transformer/output_layer_norm/gamma",
        }


class ElectraCheckpointConverter(BertCheckpointConverter):
    @property
    def original_root_name(self) -> str:
        return "electra"

    @property
    def exclude_keywords(self) -> set[str]:
        return {*super().exclude_keywords, "generator", "discriminator_predictions"}