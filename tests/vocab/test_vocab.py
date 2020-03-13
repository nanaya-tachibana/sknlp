import json
from collections import Counter
from sknlp.vocab import Vocab


class TestVocab:

    counter = Counter({'a': 100, 'b': 10, 'c': 4, 'd': 5, '<unk>': 200})

    def test_empty_init(self):
        vocab = Vocab(unk_token='unk', padding_token='pad',
                      bos_token='bos', eos_token='eos')
        assert vocab.unk == 'unk'
        assert vocab.pad == 'pad'
        assert vocab.bos == 'bos'
        assert vocab.eos == 'eos'
        assert len(vocab) == 4
        assert 'eos' in vocab
        assert vocab._token2idx['bos'] == 2
        assert len(vocab._token_frequency) == 0

    def test_init(self):
        vocab = Vocab(self.counter, min_frequency=5)
        assert vocab['<unk>'] == 1
        assert len(vocab) == 7

    def test_token2idx(self):
        vocab = Vocab(self.counter, min_frequency=5)
        assert vocab.token2idx('a') == 4
        assert vocab.token2idx(['a', 'd']) == [4, 6]
        assert vocab['a'] == 4
        assert vocab[['a', 'd']] == [4, 6]

    def test_idx2token(self):
        vocab = Vocab(self.counter, min_frequency=5)
        assert vocab.idx2token(4) == 'a'
        assert vocab.idx2token([4, 6]) == ['a', 'd']

    def test_to_json(self):
        vocab = Vocab(self.counter, min_frequency=5)
        json_dict = json.loads(vocab.to_json())
        assert json_dict['tokens']['a'] == 100
        assert json_dict['unk'] == '<unk>'

    def test_from_json(self):
        vocab = Vocab(self.counter, min_frequency=5)
        vocab = Vocab.from_json(vocab.to_json())
        assert vocab.unk == '<unk>'
        assert vocab['a'] == 4
