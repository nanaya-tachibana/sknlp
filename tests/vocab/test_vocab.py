import json
from collections import Counter
from sknlp.vocab import Vocab


class TestVocab:

    counter = Counter({'a': 100, 'b': 10, 'c': 4, 'd': 5, '<unk>': 200})

    def test_empty_init(self):
        vocab = Vocab(unk_token='unk', pad_token='pad')
        assert vocab.unk == 'unk'
        assert vocab.pad == 'pad'
        assert len(vocab) == len(vocab._reversed_tokens)
        assert 'pad' in vocab
        assert vocab._token2idx['unk'] == 1
        assert len(vocab._token_frequency) == 0

    def test_init(self):
        vocab = Vocab(self.counter, min_frequency=5)
        assert vocab['<unk>'] == 1
        assert len(vocab) == len(vocab._reversed_tokens) + 3

    def test_token2idx(self):
        vocab = Vocab(self.counter, min_frequency=5)
        n_reversed = len(vocab._reversed_tokens)
        assert vocab.token2idx('a') == n_reversed + 0
        assert vocab.token2idx(['a', 'd']) == [n_reversed + 0, n_reversed + 2]
        assert vocab['a'] == n_reversed + 0
        assert vocab[['a', 'd']] == [n_reversed + 0, n_reversed + 2]

    def test_idx2token(self):
        vocab = Vocab(self.counter, min_frequency=5)
        n_reversed = len(vocab._reversed_tokens)
        assert vocab.idx2token(n_reversed) == 'a'
        assert vocab.idx2token([n_reversed, n_reversed + 2]) == ['a', 'd']

    def test_to_json(self):
        vocab = Vocab(self.counter, min_frequency=5)
        json_dict = json.loads(vocab.to_json())
        assert json_dict['tokens']['a'] == 100
        assert json_dict['unk'] == '<unk>'

    def test_from_json(self):
        vocab = Vocab(self.counter, min_frequency=5)
        vocab = Vocab.from_json(vocab.to_json())
        assert vocab.unk == '<unk>'
        assert vocab['a'] == len(vocab._reversed_tokens) + 0
