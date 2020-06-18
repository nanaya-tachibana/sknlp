import tensorflow as tf


class BertTokenizationLayer(tf.keras.layers.Layer):
    """
    The convention in BERT is:
    (a) For sequence pairs:
    tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    (b) For single sequences:
    tokens:   [CLS] the dog is hairy . [SEP]
    type_ids: 0     0   0   0  0     0 0

    Where "type_ids" are used to indicate whether this is the first
    sequence or the second sequence. The embedding vectors for `type=0` and
    `type=1` were learned during pre-training and are added to the wordpiece
    embedding vector (and position vector). This is not *strictly* necessary
    since the [SEP] token unambiguously separates the sequences, but it makes
    it easier for the model to learn the concept of sequences.

    For classification tasks, the first vector (corresponding to [CLS]) is
    used as the "sentence vector". Note that this only makes sense because
    the entire model is fine-tuned.
    See more details in https://github.com/tensorflow/blob/master/official/nlp/data/classifier_data_lib.py.
    """
    def __init__(self, sep="@!@", name="bert_tokenization", **kwargs):
        self.sep = sep
        super().__init__(name=name, **kwargs)

    def call(self, inputs):

        def padding_string(x):
            return (
                tf.constant(["[CLS]" + self.sep])
                + x
                + tf.constant([self.sep + "[SEP]"])
            )

        padded_inputs = tf.ragged.map_flat_values(padding_string, inputs)
        r = tf.strings.split(padded_inputs, sep=self.sep)
        return tf.concat(
            [
                r[:, :1],
                tf.strings.unicode_split(tf.squeeze(r[:, 1:-1], axis=1), "UTF-8"),
                r[:, -1:]
            ],
            axis=1
        )

    def get_config(self):
        return {**super().get_config(), "sep": self.sep}
