from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.backend as K
from sknlp.module.text_rnn import TextRNN
from .deep_classifier import DeepClassifier


class TextRCNNClassifier(DeepClassifier, TextRNN):

    def __init__(self,
                 classes,
                 is_multilabel=True,
                 segmenter='jieba',
                 max_length=100,
                 embed_size=100,
                 num_rnn_layers=1,
                 rnn_hidden_size=512,
                 rnn_projection_size=128,
                 rnn_recurrent_clip=3,
                 rnn_projection_clip=3,
                 rnn_input_dropout=0.5,
                 rnn_recurrent_dropout=0.5,
                 rnn_output_dropout=0.5,
                 num_fc_layers=2,
                 fc_hidden_size=128,
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
                 vocab=None,
                 token2vec=None,
                 algorithm='text_rcnn',
                 **kwargs):
        super().__init__(classes,
                         is_multilabel=is_multilabel,
                         segmenter=segmenter,
                         max_length=max_length,
                         embed_size=embed_size,
                         algorithm=algorithm,
                         num_rnn_layers=num_rnn_layers,
                         rnn_hidden_size=rnn_hidden_size,
                         rnn_projection_size=rnn_projection_size,
                         rnn_recurrent_clip=rnn_recurrent_clip,
                         rnn_projection_clip=rnn_projection_clip,
                         rnn_input_dropout=rnn_input_dropout,
                         rnn_recurrent_dropout=rnn_recurrent_dropout,
                         rnn_output_dropout=rnn_output_dropout,
                         rnn_last_connection='all',
                         num_fc_layers=num_fc_layers,
                         fc_hidden_size=fc_hidden_size,
                         rnn_kernel_initializer=rnn_kernel_initializer,
                         rnn_recurrent_initializer=rnn_recurrent_initializer,
                         rnn_projection_initializer=rnn_projection_initializer,
                         rnn_bias_initializer=rnn_bias_initializer,
                         rnn_kernel_regularizer=rnn_kernel_regularizer,
                         rnn_recurrent_regularizer=rnn_recurrent_regularizer,
                         rnn_projection_regularizer=rnn_projection_regularizer,
                         rnn_bias_regularizer=rnn_bias_regularizer,
                         rnn_kernel_constraint=rnn_kernel_constraint,
                         rnn_recurrent_constraint=rnn_recurrent_constraint,
                         rnn_projection_constraint=rnn_projection_constraint,
                         rnn_bias_constraint=rnn_bias_constraint,
                         vocab=vocab,
                         token2vec=token2vec,
                         **kwargs)

    def build_encode_layer(self, inputs):
        rnn_outputs = super().build_encode_layer(inputs)
        if self._rnn_input_dropout:
            noise_shape = (None, 1, self._embed_size)
            inputs = Dropout(self._rnn_input_dropout,
                             noise_shape=noise_shape,
                             name='embed_dropout')(inputs)
        mixed_inputs = K.concatenate([inputs, rnn_outputs], axis=-1)
        mixed_outputs = Dense(self._fc_hidden_size,
                              activation='tanh')(mixed_inputs)
        return K.max(mixed_outputs, axis=1)

    def get_config(self):
        base_config = super().get_config()
        del base_config['rnn_last_connection']
        return base_config
