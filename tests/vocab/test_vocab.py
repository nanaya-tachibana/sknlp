import pytest

from sknlp.vocab import Vocab


@pytest.fixture
def tokens():
    return ["a", "b", "c", "d", "<unk>", "<pad>", "<bos>", "<eos>"]


@pytest.fixture
def frequencies():
    return [100, 10, 4, 5, 1, 1, 1, 1]


def test_empty_vocab():
    vocab = Vocab([])
    assert vocab.pad == "<pad>"
    assert vocab.unk == "<unk>"
    assert vocab.bos == "<bos>"
    assert vocab.eos == "<eos>"
    assert vocab[vocab.pad] == 0
    assert vocab[vocab.unk] == 1
    assert vocab[vocab.bos] == 2
    assert vocab[vocab.eos] == 3


def test_vocab_without_special_token(tokens):
    vocab = Vocab(tokens[:-4])
    assert len(vocab) == len(tokens)
    assert vocab.pad == "<pad>"


def test_vocab_with_special_token(tokens, frequencies):
    vocab = Vocab(
        tokens,
        frequencies=frequencies,
        min_frequency=5,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    assert len(vocab) == len(tokens) - 1
    assert "c" not in vocab
    assert "b" in vocab
    assert vocab["a"] == 2
    assert vocab.token2idx(["a", "b", "<unk>"]) == [2, 3, 5]
    assert vocab.idx2token([2, 3, 5]) == ["a", "b", "<unk>"]
    assert vocab.sorted_tokens == [
        "<s>",
        "</s>",
        "a",
        "b",
        "d",
        "<unk>",
        "<pad>",
    ]


def test_vocab_serialization(tokens, frequencies):
    vocab = Vocab(
        tokens,
        frequencies,
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