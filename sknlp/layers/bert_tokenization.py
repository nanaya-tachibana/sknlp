import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="sknlp")
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

    The mask has 1 for real tokens and 0 for padding tokens. Only real
    tokens are attended to.
    See more details in https://github.com/tensorflow/models/blob/ad423d065701785587d13d0fe7b566191e7378c6/official/nlp/data/classifier_data_lib.py#L668
    """

    def __init__(
        self,
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        name: str = "bert_tokenization",
        **kwargs
    ) -> None:
        self.cls_token = cls_token
        self.sep_token = sep_token
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        tokens = tf.strings.unicode_split(inputs, "UTF-8")
        cls_tokens = tf.reshape(
            tf.tile(tf.constant([self.cls_token], dtype=tf.string), [tokens.nrows()]),
            [tokens.nrows(), 1],
        )
        sep_tokens = tf.reshape(
            tf.tile(tf.constant([self.sep_token], dtype=tf.string), [tokens.nrows()]),
            [tokens.nrows(), 1],
        )
        return tf.concat([cls_tokens, tokens, sep_tokens], axis=1)

    def get_config(self):
        return {
            **super().get_config(),
            "sep_token": self.sep_token,
            "cls_token": self.cls_token,
        }
