from typing import List

from tensorflow.keras.layers import Bidirectional, Dropout

from sknlp.layers import MLPLayer, LSTMP

from .supervised_model import SupervisedNLPModel


class TextRNN(SupervisedNLPModel):

    def __init__(self,
                 classes,
                 segmenter='jieba',
                 max_sequence_length=80,
                 embedding_size=100,
                 num_rnn_layers=1,
                 rnn_hidden_size=256,
                 rnn_projection_size=128,
                 rnn_recurrent_clip=3,
                 rnn_projection_clip=3,
                 rnn_input_dropout=0.5,
                 rnn_recurrent_dropout=0.5,
                 rnn_output_dropout=0.5,
                 rnn_last_connection=None,
                 num_fc_layers=2,
                 fc_hidden_size=256,
                 rnn_kernel_initializer='glorot_uniform',
                 rnn_recurrent_initializer='orthogonal',
                 rnn_projection_initializer='glorot_uniform',
                 rnn_bias_initializer='zeros',
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
                         segmenter=segmenter,
                         embedding_size=embedding_size,
                         max_sequence_length=max_sequence_length,
                         text2vec=text2vec,
                         **kwargs)
        self._num_rnn_layers = num_rnn_layers
        self._rnn_hidden_size = rnn_hidden_size
        self._rnn_last_connection = rnn_last_connection
        self._rnn_projection_size = rnn_projection_size
        self._rnn_recurrent_clip = rnn_recurrent_clip
        self._rnn_projection_clip = rnn_projection_clip
        self._rnn_input_dropout = rnn_input_dropout
        self._rnn_recurrent_dropout = rnn_recurrent_dropout
        self._rnn_output_dropout = rnn_output_dropout
        self._rnn_kernel_initializer = rnn_kernel_initializer
        self._rnn_recurrent_initializer = rnn_recurrent_initializer
        self._rnn_projection_initializer = rnn_projection_initializer
        self._rnn_bias_initializer = rnn_bias_initializer
        self._rnn_kernel_regularizer = rnn_kernel_regularizer
        self._rnn_recurrent_regularizer = rnn_recurrent_regularizer
        self._rnn_projection_regularizer = rnn_projection_regularizer
        self._rnn_bias_regularizer = rnn_bias_regularizer
        self._rnn_kernel_constraint = rnn_kernel_constraint
        self._rnn_recurrent_constraint = rnn_recurrent_constraint
        self._rnn_projection_constraint = rnn_projection_constraint
        self._rnn_bias_constraint = rnn_bias_constraint
        self._num_fc_layers = num_fc_layers
        self._fc_hidden_size = fc_hidden_size

    def build_encode_layer(self, inputs):
        for i in range(self._num_rnn_layers):
            return_sequences = (
                i != self._num_rnn_layers - 1
                or self._rnn_last_connection != 'last'
            )
            rnn_outputs = Bidirectional(LSTMP(
                self._rnn_hidden_size,
                self._rnn_projection_size,
                kernel_initializer=self._rnn_kernel_initializer,
                recurrent_initializer=self._rnn_recurrent_initializer,
                projection_initializer=self._rnn_projection_initializer,
                bias_initializer=self._rnn_bias_initializer,
                kernel_regularizer=self._rnn_kernel_regularizer,
                recurrent_regularizer=self._rnn_recurrent_regularizer,
                projection_regularizer=self._rnn_projection_regularizer,
                bias_regularizer=self._rnn_bias_regularizer,
                kernel_constraint=self._rnn_kernel_constraint,
                recurrent_constraint=self._rnn_recurrent_constraint,
                projection_constraint=self._rnn_projection_constraint,
                bias_constraint=self._rnn_bias_constraint,
                dropout=self._rnn_input_dropout,
                recurrent_dropout=self._rnn_recurrent_dropout,
                recurrent_clip=self._rnn_recurrent_clip,
                projection_clip=self._rnn_projection_clip,
                return_sequences=return_sequences
            ))(inputs)
            inputs = rnn_outputs
        if self._rnn_output_dropout:
            noise_shape = (None, 1, self._rnn_projection_size * 2)
            if self._rnn_last_connection == 'last':
                noise_shape = None
            return Dropout(self._rnn_output_dropout,
                           noise_shape=noise_shape,
                           name='rnn_output_dropout')(inputs)
        else:
            return inputs

    def build_output_layer(self, inputs):
        mlp = MLPLayer(self._num_fc_layers,
                       hidden_size=self._fc_hidden_size,
                       output_size=self._num_classes)
        return mlp(inputs)

    @property
    def output_names(self) -> List[str]:
        return ["mlp"]

    @property
    def output_types(self) -> List[str]:
        return ["float"]

    @property
    def output_shapes(self) -> List[List[int]]:
        return [[-1, self._num_classes]]

    def get_config(self):
        return {
            **super().get_config(),
            'num_rnn_layers': self._num_rnn_layers,
            'rnn_hidden_size': self._rnn_hidden_size,
            'rnn_projection_size': self._rnn_projection_size,
            'rnn_recurrent_clip': self._rnn_recurrent_clip,
            'rnn_projection_clip': self._rnn_projection_clip,
            'rnn_input_dropout': self._rnn_input_dropout,
            'rnn_recurrent_dropout': self._rnn_recurrent_dropout,
            'rnn_output_dropout': self._rnn_output_dropout,
            'rnn_last_connection': self._rnn_last_connection,
            'num_fc_layers': self._num_fc_layers,
            'fc_hidden_size': self._fc_hidden_size,
            'rnn_kernel_initializer': self._rnn_kernel_initializer,
            'rnn_recurrent_initializer': self._rnn_recurrent_initializer,
            'rnn_projection_initializer': self._rnn_projection_initializer,
            'rnn_bias_initializer': self._rnn_bias_initializer,
            'rnn_kernel_regularizer': self._rnn_kernel_regularizer,
            'rnn_recurrent_regularizer': self._rnn_recurrent_regularizer,
            'rnn_projection_regularizer': self._rnn_projection_regularizer,
            'rnn_bias_regularizer': self._rnn_bias_regularizer,
            'rnn_kernel_constraint': self._rnn_kernel_constraint,
            'rnn_recurrent_constraint': self._rnn_recurrent_constraint,
            'rnn_projection_constraint': self._rnn_projection_constraint,
            'rnn_bias_constraint': self._rnn_bias_constraint
        }

    def get_custom_objects(self):
        return {**super().get_custom_objects(),
                'LSTMP': LSTMP,
                'MLPLayer': MLPLayer}
