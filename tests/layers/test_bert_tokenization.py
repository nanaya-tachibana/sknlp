import pytest

import tensorflow as tf

from sknlp.layers import BertTokenizationLayer


@pytest.mark.parametrize(
    "input,expected",
    [
        ("寄快递", ["[CLS]", "寄", "快", "递", "[SEP]"]),
        ("是的", ["[CLS]", "是", "的", "[SEP]"]),
        ("hello world", ["[CLS]", "h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d", "[SEP]"])
    ]
)
def test_bert_tokenization_layer(input, expected):
    layer = BertTokenizationLayer()
    sentence_tokens = [
        token.decode("UTF-8")
        for token in layer(tf.constant(input)).numpy().tolist()[0]
    ]
    assert sentence_tokens == expected
