from __future__ import annotations
from typing import Optional, Union, Sequence

import json
from collections import Counter


class Vocab:
    def __init__(
        self,
        counter: Optional[Counter] = None,
        min_frequency: int = 1,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
    ) -> None:
        self._min_frequency = min_frequency
        self._unk_token = unk_token
        self._pad_token = pad_token
        self._bos_token = bos_token
        self._eos_token = eos_token

        self._token2idx = dict()

        self._token_frequency = dict()
        if counter is None or pad_token not in counter:
            self._token2idx[pad_token] = len(self._token2idx)
        if counter is None or unk_token not in counter:
            self._token2idx[unk_token] = len(self._token2idx)
        if counter is None or (bos_token is not None and bos_token not in counter):
            self._token2idx[bos_token] = len(self._token2idx)
        if counter is None or (eos_token is not None and eos_token not in counter):
            self._token2idx[eos_token] = len(self._token2idx)
        if counter is not None:
            for _, (token, frequency) in enumerate(counter.most_common()):
                if frequency >= min_frequency:
                    self._token2idx[token] = len(self._token2idx)
                    self._token_frequency[token] = frequency
        self._idx2token = dict(zip(self._token2idx.values(), self._token2idx.keys()))

    def set_vocab(
        self, token2idx: dict[str, int], pad_token: str, unk_token: str
    ) -> None:
        if token2idx.get(pad_token, None) is None:
            raise ValueError("no padding token")
        if token2idx.get(unk_token, None) is None:
            raise ValueError("no unk token")
        self._token2idx = dict()
        self._token2idx.update(**token2idx)
        self._idx2token = dict(zip(self._token2idx.values(), self._token2idx.keys()))

    def idx2token(self, indices: Union[int, Sequence[int]]) -> Union[str, list[str]]:
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
        if isinstance(indices, int):
            idx = indices
            if idx not in self._idx2token:
                raise KeyError("index %d is out of vacab" % idx)
            return self._idx2token[idx]
        elif isinstance(indices, (list, tuple)):
            res = []
            for idx in indices:
                res.append(self.idx2token(idx))
            return res
        else:
            raise ValueError(
                "indices should be int or a list of int, "
                "but %s was given" % type(indices)
            )

    def token2idx(self, tokens: Union[str, list[str]]) -> Union[int, list[int]]:
        if isinstance(tokens, str):
            return self._token2idx.get(tokens, self._token2idx[self.unk])
        elif isinstance(tokens, (list, tuple)):
            return [self.token2idx(token) for token in tokens]
        else:
            raise ValueError(
                "tokens should be str or a list of str, "
                "but %s was given" % type(tokens)
            )

    def to_json(self) -> str:
        return json.dumps(
            {
                "tokens": self._token_frequency,
                "token2idx": self._token2idx,
                "unk": self.unk,
                "pad": self.pad,
                "bos": self.bos,
                "eos": self.eos,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Vocab":
        vocab_dict = json.loads(json_str)
        vocab = cls(
            min_frequency=1,
            pad_token=vocab_dict["pad"],
            unk_token=vocab_dict["unk"],
            bos_token=vocab_dict.get("bos", None),
            eos_token=vocab_dict.get("eos", None),
        )
        vocab.set_vocab(vocab_dict["token2idx"], vocab_dict["pad"], vocab_dict["unk"])
        return vocab

    def __getitem__(self, tokens: Union[str, Sequence[str]]) -> list[int]:
        return self.token2idx(tokens)

    def __contains__(self, token: str) -> bool:
        return token in self._token2idx

    def __len__(self) -> int:
        return len(self._token2idx)

    def __repr__(self) -> str:
        return 'Vocab(size=%d, unk="%s")' % (len(self), self._unk_token)

    @property
    def pad(self) -> str:
        return self._pad_token

    @property
    def unk(self) -> str:
        return self._unk_token

    @property
    def bos(self) -> str:
        return self._bos_token

    @property
    def eos(self) -> str:
        return self._eos_token

    @property
    def sorted_tokens(self) -> list[str]:
        items = sorted(self._token2idx.items(), key=lambda x: x[1])
        return [k for k, _ in items]