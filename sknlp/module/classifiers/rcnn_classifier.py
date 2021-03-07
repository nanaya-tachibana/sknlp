from typing import List

from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

from sknlp.layers import MultiLSTMP, MLPLayer, LSTMP, LSTMPCell
from .deep_classifier import DeepClassifier


class TextRCNNClassifier(DeepClassifier):
    def __init__(
        self,
        classes: List[str],
        is_multilabel: bool = True,
        max_sequence_length: int = 100,
        sequence_length: int = None,
        segmenter: str = "jieba",
        embedding_size: int = 100,
        use_batch_normalization: bool = True,
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
        fc_activation: str = "tanh",
        fc_momentum: float = 0.9,
        fc_epsilon: float = 1e-5,
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
            is_multilabel=is_multilabel,
            max_sequence_length=max_sequence_length,
            sequence_length=sequence_length,
            segmenter=segmenter,
            embedding_size=embedding_size,
            use_batch_normalization=use_batch_normalization,
            algorithm="rcnn",
            text2vec=text2vec,
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
        self.fc_activation = fc_activation
        self.fc_momentum = fc_momentum
        self.fc_epsilon = fc_epsilon

    def build_encode_layer(self, inputs):
        rnn_outputs = MultiLSTMP(
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
        )(self.text2vec(inputs))
        if self.rnn_layer.output_dropout:
            noise_shape = (None, 1, self._text2vec.embedding_size)
            inputs = Dropout(
                self.rnn_layer.output_dropout,
                noise_shape=noise_shape,
                name="embedding_dropout",
            )(inputs)
        mixed_inputs = tf.concatenate([inputs, rnn_outputs], axis=-1)
        mixed_outputs = Dense(self.mlp_layer.hidden_size, activation="tanh")(
            mixed_inputs
        )
        return tf.math.max(mixed_outputs, axis=1)

    def build_output_layer(self, inputs):
        return MLPLayer(
            self.num_fc_layers,
            hidden_size=self.fc_hidden_size,
            output_size=self.num_classes,
            activation=self.fc_activation,
            batch_normalization=self.use_batch_normalization,
            momentum=self.fc_momentum,
            epsilon=self.fc_epsilon,
            name="mlp",
        )(inputs)

    def get_custom_objects(self):
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
