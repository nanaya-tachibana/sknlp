from __future__ import annotations
from typing import Optional, Union, Sequence

import json
from collections import Counter

DEFAULT_SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class Vocab:
    def __init__(
        self,
        counter: Optional[Counter] = None,
        min_frequency: int = 1,
        pad_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
    ) -> None:
        if counter is not None and len(counter) < 4:
            raise ValueError(f"`counter`大小至少为4, 实际为{len(counter)}.")
        pad_id = 0
        unk_id = 1
        bos_id = 2
        eos_id = 3

        self._min_frequency = min_frequency

        self._token2idx = dict()
        self._token_frequency = dict()
        if counter is None:
            counter = Counter(dict(zip(DEFAULT_SPECIAL_TOKENS, [1] * 4)))

        for token, frequency in counter.most_common():
            is_special_token = True
            if token == pad_token:
                pad_id = len(self._token2idx)
            elif token == unk_token:
                unk_id = len(self._token2idx)
            elif token == bos_token:
                bos_id = len(self._token2idx)
            elif token == eos_token:
                eos_id = len(self._token2idx)
            else:
                is_special_token = False
            if is_special_token or frequency >= min_frequency:
                self._token2idx[token] = len(self._token2idx)
                self._token_frequency[token] = frequency
        self._idx2token = dict(zip(self._token2idx.values(), self._token2idx.keys()))
        self._pad_token = self._idx2token[pad_id]
        self._unk_token = self._idx2token[unk_id]
        self._bos_token = self._idx2token[bos_id]
        self._eos_token = self._idx2token[eos_id]

    def set_vocab(
        self,
        token2idx: dict[str, int],
        pad_token: str,
        unk_token: str,
        bos_token: str,
        eos_token: str,
    ) -> None:
        if pad_token not in token2idx:
            raise ValueError(f"{pad_token} not in `token2idx`")
        if unk_token not in token2idx:
            raise ValueError(f"{unk_token} not in `token2idx`")
        if bos_token not in token2idx:
            raise ValueError(f"{bos_token} not in `token2idx`")
        if eos_token not in token2idx:
            raise ValueError(f"{eos_token} not in `token2idx`")
        self._pad_token = pad_token
        self._unk_token = unk_token
        self._bos_token = bos_token
        self._eos_token = eos_token

        self._token2idx = dict(**token2idx)
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
                "token_frequency": self._token_frequency,
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
        pad_token = vocab_dict.get("pad", None)
        unk_token = vocab_dict.get("unk", None)
        bos_token = vocab_dict.get("bos", None)
        eos_token = vocab_dict.get("eos", None)
        vocab = cls(
            Counter(**vocab_dict["token_frequency"]),
            min_frequency=1,
        )
        vocab.set_vocab(
            vocab_dict["token2idx"], pad_token, unk_token, bos_token, eos_token
        )
        return vocab

    def __getitem__(self, tokens: Union[str, Sequence[str]]) -> list[int]:
        return self.token2idx(tokens)

    def __contains__(self, token: str) -> bool:
        return token in self._token2idx

    def __len__(self) -> int:
        return len(self._token2idx)

    def __repr__(self) -> str:
        return (
            f"Vocab(size=%d, "
            f'pad="{self.pad}", '
            f'unk="{self.unk}", '
            f'bos="{self.bos}", '
            f'eos="{self.eos}")'
        )

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