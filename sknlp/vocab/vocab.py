import json
from collections import defaultdict, Counter


class Vocab:

    def __init__(self, counter=None, min_frequency=1,
                 unk_token='<unk>', padding_token='<pad>',
                 bos_token='<bos>', eos_token='<eos>'):
        self._min_frequency = min_frequency
        self._unk_token = unk_token
        self._padding_token = padding_token
        self._bos_token = bos_token
        self._eos_token = eos_token

        self._token2idx = defaultdict(lambda: 1)
        self._token2idx[padding_token] = 0
        self._token2idx[unk_token] = 1
        self._token2idx[bos_token] = 2
        self._token2idx[eos_token] = 3
        reversed_tokens = {unk_token, padding_token, bos_token, eos_token}

        self._token_frequency = defaultdict(lambda: 0)
        if counter is not None:
            for _, (token, frequency) in enumerate(counter.most_common()):
                if token not in reversed_tokens and frequency >= min_frequency:
                    self._token2idx[token] = len(self._token2idx)
                    self._token_frequency[token] = frequency
        self._idx2token = dict(zip(self._token2idx.values(),
                                   self._token2idx.keys()))

    def idx2token(self, indices):
        """
        Lookup tokens by indices.

        raise KeyError if any indices are greater than or equal to
        size of vacab.

        Parameters
        ----------
        indices: int or list of int

        Returns
        ----------
        str or list of str
        """
        if not isinstance(indices, (list, tuple)):
            if indices not in self._idx2token:
                raise(KeyError('index %d is out of vacab' % indices))
            return self._idx2token[indices]
        else:
            res = []
            for idx in indices:
                if idx not in self._idx2token:
                    raise(KeyError('index %d is out of vacab' % idx))
                res.append(self.idx2token(idx))
            return res

    def token2idx(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self._token2idx[tokens]
        else:
            return [self.token2idx(t) for t in tokens]

    def to_json(self):
        return json.dumps(
            {
                'tokens': self._token_frequency,
                'token2idx': self._token2idx,
                'unk': self._unk_token,
                'pad': self._padding_token,
                'bos': self._bos_token,
                'eos': self._eos_token
            },
            ensure_ascii=False
        )

    @classmethod
    def from_json(cls, json_str):
        vocab_dict = json.loads(json_str)
        return cls(Counter(vocab_dict['tokens']),
                   min_frequency=1,
                   unk_token=vocab_dict['unk'],
                   padding_token=vocab_dict['pad'],
                   bos_token=vocab_dict['bos'],
                   eos_token=vocab_dict['eos'])

    def __getitem__(self, tokens):
        return self.token2idx(tokens)

    def __contains__(self, token):
        return token in self._token2idx

    def __len__(self):
        return len(self._token2idx)

    def __repr__(self):
        return 'Vocab(size=%d, unk="%s")' % (len(self), self._unk_token)

    @property
    def pad(self):
        return self._padding_token

    @property
    def unk(self):
        return self._unk_token

    @property
    def bos(self):
        return self._bos_token

    @property
    def eos(self):
        return self._eos_token
