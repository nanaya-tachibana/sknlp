import pytest

from sknlp.data.tokenizer import BertTokenizer


@pytest.mark.parametrize(
    "max_length",
    [
        pytest.param(1, id="string length > max length"),
        pytest.param(4, id="string length <= max length"),
        pytest.param(None, id="not truncating"),
    ],
)
def test_bert_tokenizer_single_input_truncating(max_length, vocab):
    tokenizer = BertTokenizer(vocab, max_length=max_length)
    text = "甲乙丙丁"
    tokens = tokenizer.tokenize(text)
    assert len(tokens) == min(max_length or 9999999, len(text))
    for i in range(len(tokens)):
        assert text[i] == tokens[i]


@pytest.mark.parametrize(
    "max_length",
    [
        pytest.param(1, id="string length > max length"),
        pytest.param(4, id="string length <= max length"),
        pytest.param(None, id="not truncating"),
    ],
)
def test_bert_tokenizer_pair_inputs_truncating(max_length, vocab):
    tokenizer = BertTokenizer(vocab, max_length=max_length)
    texts = ["甲乙丙丁", "甲乙丙丁葵"]
    tokens_list = tokenizer.tokenize(texts)
    max_length = max_length or 9999999
    f_len = min(len(texts[0]), max_length)
    s_len = min(len(texts[1]), max_length)
    assert len(tokens_list) == 2
    assert len(tokens_list[0]) == f_len
    assert len(tokens_list[1]) == s_len
    for i in range(f_len):
        assert texts[0][i] == tokens_list[0][i]
    for i in range(s_len):
        assert texts[1][i] == tokens_list[1][i]