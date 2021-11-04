from __future__ import annotations
from typing import Optional, Sequence, Any
import json
from collections import defaultdict
from itertools import accumulate


class Vocab:
    def __init__(
        self,
        tokens: Sequence[str],
        frequencies: Optional[list[int]] = None,
        min_frequency: int = 1,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
    ) -> None:
        frequencies = frequencies or [min_frequency for _ in range(len(tokens))]
        special_tokens = (pad_token, unk_token, bos_token, eos_token)
        special_token_index_offset = 0
        _tokens = list(tokens)
        for special_token in (pad_token, unk_token, bos_token, eos_token):
            if special_token not in _tokens:
                _tokens.insert(special_token_index_offset, special_token)
                frequencies.insert(special_token_index_offset, min_frequency)
                special_token_index_offset += 1

        self._token2idx = dict()
        self._token_frequency = dict()
        for token, frequency in zip(_tokens, frequencies):
            if token in special_tokens or frequency >= min_frequency:
                self._token2idx[token] = len(self._token2idx)
                self._token_frequency[token] = frequency
        self._idx2token = dict(zip(self._token2idx.values(), self._token2idx.keys()))
        self._pad_token = pad_token
        self._unk_token = unk_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self.special_tokens = {self.pad, self.unk, self.bos, self.eos}

    def idx2token(self, indices: int | Sequence[int]) -> str | list[str]:
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

    def token2idx(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self._token2idx.get(tokens, self._token2idx[self.unk])
        elif isinstance(tokens, (list, tuple)):
            return [self.token2idx(token) for token in tokens]
        else:
            raise ValueError(
                "tokens should be str or a list of str, "
                "but %s was given" % type(tokens)
            )

    def get_token_length(self, token: str) -> int:
        length = len(token)
        if token in self.special_tokens:
            length = 1
        elif token.startswith("##"):
            length -= 2
        return length

    def create_ichar2itoken_mapping(
        self,
        tokens: Sequence[str],
    ) -> tuple[dict[int, int], dict[int, int]]:
        lengths = [self.get_token_length(token) for token in tokens]
        start_mapping: dict[int, int] = defaultdict(lambda: -1)
        end_mapping: dict[int, int] = defaultdict(lambda: -1)
        offset = -1
        for idx, length in enumerate(lengths):
            for i in range(length):
                offset += 1
                if i == 0:
                    start_mapping[offset] = idx
                if i == length - 1:
                    end_mapping[offset] = idx
        return start_mapping, end_mapping

    def create_itoken2ichar_mapping(
        self,
        tokens: Sequence[str],
    ) -> tuple[dict[int, int], dict[int, int]]:
        lengths = [self.get_token_length(token) for token in tokens]
        cumsum = list(accumulate(lengths))
        start_mapping = {i: c - l for i, (c, l) in enumerate(zip(cumsum, lengths))}
        end_mapping = {i: c - 1 for i, c in enumerate(cumsum)}
        return start_mapping, end_mapping

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
        vocab_dict: dict[str, Any] = json.loads(json_str)
        pad_token = vocab_dict["pad"]
        unk_token = vocab_dict["unk"]
        bos_token = vocab_dict["bos"]
        eos_token = vocab_dict["eos"]
        vocab = cls(
            vocab_dict["token_frequency"].keys(),
            vocab_dict["token_frequency"].values(),
            min_frequency=1,
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )
        return vocab

    def __getitem__(self, tokens: str | Sequence[str]) -> int | list[int]:
        return self.token2idx(tokens)

    def __contains__(self, token: str) -> bool:
        return token in self._token2idx

    def __len__(self) -> int:
        return len(self._token2idx)

    def __repr__(self) -> str:
        return (
            f"Vocab(size={len(self)}, "
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

    @property
    def sorted_token_lengths(self) -> list[int]:
        return [self.get_token_length(token) for token in self.sorted_tokens]