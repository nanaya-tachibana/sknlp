from typing import List, Dict, Any

import tensorflow as tf

from sknlp.layers import MultiLSTMP, MLPLayer, LSTMP, LSTMPCell, CrfLossLayer
from .deep_tagger import DeepTagger


class TextRNNTagger(DeepTagger):
    def __init__(
        self,
        classes: List[str],
        max_sequence_length: int = 100,
        sequence_length: int = None,
        segmenter: str = "char",
        embedding_size: int = 100,
        num_rnn_layers: int = 1,
        rnn_hidden_size: int = 512,
        rnn_projection_size: int = 128,
        rnn_recurrent_clip: float = 3.0,
        rnn_projection_clip: float = 3.0,
        rnn_input_dropout: float = 0.5,
        rnn_recurrent_dropout: float = 0.5,
        rnn_output_dropout: float = 0.5,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 128,
        rnn_kernel_initializer="glorot_uniform",
        rnn_recurrent_initializer="orthogonal",
        rnn_projection_initializer="glorot_uniform",
        rnn_bias_initializer="zeros",
        rnn_kernel_regularizer=None,
        rnn_recurrent_regularizer=None,
        rnn_projection_regularizer=None,
        rnn_bias_regularizer=None,
        rnn_kernel_constraint=None,
        rnn_recurrent_constraint=None,
        rnn_projection_constraint=None,
        rnn_bias_constraint=None,
        text2vec=None,
        **kwargs
    ):
        super().__init__(
            classes,
            start_tag=None,
            end_tag=None,
            max_sequence_length=max_sequence_length,
            sequence_length=sequence_length,
            segmenter=segmenter,
            embedding_size=embedding_size,
            text2vec=text2vec,
            algorithm="rnn",
            **kwargs
        )
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_projection_size = rnn_projection_size
        self.rnn_recurrent_clip = rnn_recurrent_clip
        self.rnn_projection_clip = rnn_projection_clip
        self.rnn_input_dropout = rnn_input_dropout
        self.rnn_recurrent_dropout = rnn_recurrent_dropout
        self.rnn_output_dropout = rnn_output_dropout
        self.rnn_kernel_initializer = rnn_kernel_initializer
        self.rnn_recurrent_initializer = rnn_recurrent_initializer
        self.rnn_projection_initializer = rnn_projection_initializer
        self.rnn_bias_initializer = rnn_bias_initializer
        self.rnn_kernel_regularizer = rnn_kernel_regularizer
        self.rnn_recurrent_regularizer = rnn_recurrent_regularizer
        self.rnn_projection_regularizer = rnn_projection_regularizer
        self.rnn_bias_regularizer = rnn_bias_regularizer
        self.rnn_kernel_constraint = rnn_kernel_constraint
        self.rnn_recurrent_constraint = rnn_recurrent_constraint
        self.rnn_projection_constraint = rnn_projection_constraint
        self.rnn_bias_constraint = rnn_bias_constraint
        self.num_fc_layers = num_fc_layers
        self.fc_hidden_size = fc_hidden_size
        self.inputs = [
            tf.keras.Input(shape=(None,), dtype=tf.int32, name="token_id"),
            tf.keras.Input(shape=(None,), dtype=tf.int32, name="tag_id"),
        ]

    def build_encode_layer(self, inputs):
        token_ids, tag_ids = inputs
        mask = tf.keras.layers.Lambda(
            lambda x: tf.cast(x != 0, tf.int32), name="mask_layer"
        )(token_ids)
        return (
            MultiLSTMP(
                self.num_rnn_layers,
                self.rnn_hidden_size,
                projection_size=self.rnn_projection_size,
                recurrent_clip=self.rnn_recurrent_clip,
                projection_clip=self.rnn_projection_clip,
                input_dropout=self.rnn_input_dropout,
                recurrent_dropout=self.rnn_recurrent_dropout,
                output_dropout=self.rnn_output_dropout,
                last_connection="all",
                kernel_initializer=self.rnn_kernel_initializer,
                recurrent_initializer=self.rnn_recurrent_initializer,
                projection_initializer=self.rnn_projection_initializer,
                bias_initializer=self.rnn_bias_initializer,
                kernel_regularizer=self.rnn_kernel_regularizer,
                recurrent_regularizer=self.rnn_recurrent_regularizer,
                projection_regularizer=self.rnn_projection_regularizer,
                bias_regularizer=self.rnn_bias_regularizer,
                kernel_constraint=self.rnn_kernel_constraint,
                recurrent_constraint=self.rnn_recurrent_constraint,
                projection_constraint=self.rnn_projection_constraint,
                bias_constraint=self.rnn_bias_constraint,
                name="rnn",
            )(self.text2vec(token_ids)),
            mask,
            tag_ids,
        )

    def build_output_layer(self, inputs):
        embeddings, mask, tag_ids = inputs
        emissions = MLPLayer(
            self.num_fc_layers,
            hidden_size=self.fc_hidden_size,
            output_size=self.num_classes,
            name="mlp",
        )(embeddings)
        return CrfLossLayer(self.num_classes)([emissions, tag_ids], mask)

    @classmethod
    def _filter_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        config = super()._filter_config(config)
        config.pop("start_tag", None)
        config.pop("end_tag", None)
        return config

    def get_custom_objects(self) -> Dict[str, Any]:
        return {
            **super().get_custom_objects(),
            "MLPLayer": MLPLayer,
            "LSTMPCell": LSTMPCell,
            "LSTMP": LSTMP,
            "MultiLSTMP": MultiLSTMP,
            "GlorotUniform": tf.keras.initializers.GlorotUniform,
            "Orthogonal": tf.keras.initializers.Orthogonal,
            "Zeros": tf.keras.initializers.Zeros,
        }