from typing import List

from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.backend as K

from sknlp.layers import MultiLSTMP, MLPLayer, LSTMP
from .deep_classifier import DeepClassifier


class TextRCNNClassifier(DeepClassifier):

    def __init__(self,
                 classes: List[str],
                 is_multilabel: bool = True,
                 max_sequence_length: int = 100,
                 sequence_length: int = None,
                 segmenter: str = "jieba",
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
                 **kwargs):
        super().__init__(classes,
                         is_multilabel=is_multilabel,
                         max_sequence_length=max_sequence_length,
                         sequence_length=sequence_length,
                         segmenter=segmenter,
                         embedding_size=embedding_size,
                         algorithm="text_rnn",
                         **kwargs)
        self.rnn_layer = MultiLSTMP(
            num_rnn_layers,
            rnn_hidden_size,
            projection_size=rnn_projection_size,
            recurrent_clip=rnn_recurrent_clip,
            projection_clip=rnn_projection_clip,
            input_dropout=rnn_input_dropout,
            recurrent_dropout=rnn_recurrent_dropout,
            output_dropout=rnn_output_dropout,
            last_connection="all",
            kernel_initializer=rnn_kernel_initializer,
            recurrent_initializer=rnn_recurrent_initializer,
            projection_initializer=rnn_projection_initializer,
            bias_initializer=rnn_bias_initializer,
            kernel_regularizer=rnn_kernel_regularizer,
            recurrent_regularizer=rnn_recurrent_regularizer,
            projection_regularizer=rnn_projection_regularizer,
            bias_regularizer=rnn_bias_regularizer,
            kernel_constraint=rnn_kernel_constraint,
            recurrent_constraint=rnn_recurrent_constraint,
            projection_constraint=rnn_projection_constraint,
            bias_constraint=rnn_bias_constraint,
            name="rnn"
        )
        self.mlp_layer = MLPLayer(
            num_fc_layers, hidden_size=fc_hidden_size,
            output_size=self.num_classes, name="mlp"
        )

    def build_encode_layer(self, inputs):
        rnn_outputs = self.rnn_layer(inputs)
        if self.rnn_layer.output_dropout:
            noise_shape = (None, 1, self._text2vec.embedding_size)
            inputs = Dropout(self.rnn_layer.output_dropout,
                             noise_shape=noise_shape,
                             name="embedding_dropout")(inputs)
        mixed_inputs = K.concatenate([inputs, rnn_outputs], axis=-1)
        mixed_outputs = Dense(
            self.mlp_layer.hidden_size, activation="tanh"
        )(mixed_inputs)
        return K.max(mixed_outputs, axis=1)

    def build_output_layer(self, inputs):
        return self.mlp_layer(inputs)

    def get_custom_objects(self):
        return {
            **super().get_custom_objects(),
            "MultiLSTMP": MultiLSTMP,
            "MLPLayer": MLPLayer,
            "LSTMP": LSTMP
        }
