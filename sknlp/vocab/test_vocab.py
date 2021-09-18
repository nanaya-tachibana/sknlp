import pytest

from collections import Counter
from sknlp.vocab import Vocab


@pytest.fixture
def counter():
    return Counter(
        {
            "a": 100,
            "b": 10,
            "c": 4,
            "d": 5,
            "<unk>": 1,
            "<pad>": 1,
            "<bos>": 1,
            "<eos>": 1,
        }
    )


def test_empty_vocab():
    vocab = Vocab()
    assert vocab.pad == "<pad>"
    assert vocab.unk == "<unk>"
    assert vocab.bos == "<bos>"
    assert vocab.eos == "<eos>"
    assert vocab[vocab.pad] == 0
    assert vocab[vocab.unk] == 1
    assert vocab[vocab.bos] == 2
    assert vocab[vocab.eos] == 3


def test_vocab(counter):
    vocab = Vocab(
        counter=counter,
        min_frequency=5,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    assert len(vocab) == len(counter) - 1
    assert "c" not in vocab
    assert "b" in vocab
    assert vocab["a"] == 0
    assert vocab.token2idx(["a", "b", "<unk>"]) == [0, 1, 3]
    assert vocab.idx2token([0, 1, 3]) == ["a", "b", "<unk>"]
    assert vocab.sorted_tokens == [
        "a",
        "b",
        "d",
        "<unk>",
        "<pad>",
        "<bos>",
        "<eos>",
    ]


def test_vocab_serialization(counter):
    vocab = Vocab(
        counter=counter,
        min_frequency=5,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    new_vocab = Vocab.from_json(vocab.to_json())
    assert len(vocab) == len(new_vocab)
    assert vocab.sorted_tokens == new_vocab.sorted_tokens
    assert vocab.pad == new_vocab.pad
    assert vocab.unk == new_vocab.unk
    assert vocab.bos == new_vocab.bos
    assert vocab.eos == new_vocab.eos
    assert str(vocab) == str(new_vocab)


# class TestVocab:
#     def test_empty_init(self):
#         vocab = Vocab(unk_token="unk", pad_token="pad")
#         assert vocab.unk == "unk"
#         assert vocab.pad == "pad"
#         assert len(vocab) == len(vocab._reversed_tokens)
#         assert "pad" in vocab
#         assert vocab._token2idx["unk"] == 1
#         assert len(vocab._token_frequency) == 0

#     def test_init(self):
#         vocab = Vocab(self.counter, min_frequency=5)
#         assert vocab["<unk>"] == 1
#         assert len(vocab) == len(vocab._reversed_tokens) + 3

#     def test_token2idx(self):
#         vocab = Vocab(self.counter, min_frequency=5)
#         n_reversed = len(vocab._reversed_tokens)
#         assert vocab.token2idx("a") == n_reversed + 0
#         assert vocab.token2idx(["a", "d"]) == [n_reversed + 0, n_reversed + 2]
#         assert vocab["a"] == n_reversed + 0
#         assert vocab[["a", "d"]] == [n_reversed + 0, n_reversed + 2]

#     def test_sorted_tokens(self):
#         vocab = Vocab(self.counter, min_frequency=5)
#         assert vocab.sorted_tokens == ["<pad>", "<unk>", "a", "b", "d"]

#     def test_idx2token(self):
#         vocab = Vocab(self.counter, min_frequency=5)
#         n_reversed = len(vocab._reversed_tokens)
#         assert vocab.idx2token(n_reversed) == "a"
#         assert vocab.idx2token([n_reversed, n_reversed + 2]) == ["a", "d"]

#     def test_to_json(self):
#         vocab = Vocab(self.counter, min_frequency=5)
#         json_dict = json.loads(vocab.to_json())
#         assert json_dict["tokens"]["a"] == 100
#         assert json_dict["unk"] == "<unk>"

#     def test_from_json(self):
#         vocab = Vocab(self.counter, min_frequency=5)
#         vocab = Vocab.from_json(vocab.to_json())
#         assert vocab.unk == "<unk>"
#         assert vocab["a"] == len(vocab._reversed_tokens) + 0