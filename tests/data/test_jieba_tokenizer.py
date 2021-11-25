from sknlp.data.tokenizer import JiebaTokenizer
from sknlp.vocab import Vocab


def test_jieba_tokenizer():
    vocab = Vocab(["你们", "我们", "好"])
    tokenizer = JiebaTokenizer(vocab)
    tokens = tokenizer.tokenize("你们与我们好")
    assert tokens[0] == "你们"
    assert tokens[1] == "与"
    assert tokens[2] == "我们"
    assert tokens[3] == "好"