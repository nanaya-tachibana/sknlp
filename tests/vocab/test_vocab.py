import pytest

from sknlp.vocab import Vocab


@pytest.fixture
def tokens():
    return ["a", "##b", "c", "ddd", "<unk>", "<pad>", "<bos>", "<eos>"]


@pytest.fixture
def frequencies():
    return [100, 10, 4, 5, 1, 1, 1, 1]


@pytest.fixture
def input_tokens():
    return ["a", "ddd", "ccxx", "a", "##b"]


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
    assert "##b" in vocab
    assert vocab["a"] == 2
    assert vocab.token2idx(["a", "##b", "<unk>"]) == [2, 3, 5]
    assert vocab.idx2token([2, 3, 5]) == ["a", "##b", "<unk>"]
    assert vocab.sorted_tokens == [
        "<s>",
        "</s>",
        "a",
        "##b",
        "ddd",
        "<unk>",
        "<pad>",
    ]
    assert vocab.sorted_token_lengths == [1, 1, 1, 1, 3, 1, 1]


@pytest.mark.parametrize(
    "char_start,char_end,token_start,token_end",
    [
        (0, 3, 0, 1),
        (1, 3, 1, 1),
        (1, 2, 1, -1),
        (2, 3, -1, 1),
        (4, 7, 2, 2),
        (8, 9, 3, 4),
    ],
)
def test_vocab_ichar2itoken(
    char_start, char_end, token_start, token_end, input_tokens, tokens, frequencies
):
    vocab = Vocab(
        tokens,
        frequencies=frequencies,
        min_frequency=5,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    start_mapping, end_mapping = vocab.create_ichar2itoken_mapping(input_tokens)
    assert start_mapping[char_start] == token_start
    assert end_mapping[char_end] == token_end


@pytest.mark.parametrize(
    "token_start,token_end,char_start,char_end",
    [
        (0, 1, 0, 3),
        (1, 1, 1, 3),
        (1, 2, 1, 7),
        (1, 3, 1, 8),
    ],
)
def test_vocab_itoken2ichar(
    token_start, token_end, char_start, char_end, input_tokens, tokens, frequencies
):
    vocab = Vocab(
        tokens,
        frequencies=frequencies,
        min_frequency=5,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    start_mapping, end_mapping = vocab.create_itoken2ichar_mapping(input_tokens)
    assert start_mapping[token_start] == char_start
    assert end_mapping[token_end] == char_end


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