import tensorflow as tf

from sknlp.layers import BertTokenizationLayer


def test_bert_tokenization_layer():
    layer = BertTokenizationLayer()
    sentence_tokens = [
        [token.decode("UTF-8") for token in tokens.tolist()]
        for tokens in layer(tf.constant(["寄快递", "是的", "hello world"])).numpy()
    ]
    assert sentence_tokens == [
        ["[CLS]", "寄", "快", "递", "[SEP]"],
        ["[CLS]", "是", "的", "[SEP]"],
        ["[CLS]", "h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d", "[SEP]"]
    ]
