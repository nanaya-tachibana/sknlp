# Natural Language Toolkit: BLEU Score
#
# Copyright (C) 2001-2021 NLTK Project
# Authors: Chin Yee Lee, Hengfeng Li, Ruxin Hou, Calvin Tanujaya Lim
# Contributors: Björn Mattsson, Dmitrijs Milajevs, Liling Tan
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
# 测试用例来自https://github.com/nltk/nltk/blob/3.6.2/nltk/translate/bleu_score.py
from __future__ import annotations
from fractions import Fraction
from typing import Hashable, Sequence
import pytest

from sknlp.utils.generation import (
    ngrams,
    modified_precision,
    closest_reference_length,
    brevity_penalty,
    weighted_precision_sum,
    corpus_bleu,
    sentence_bleu,
)


hypothesis1 = [
    "It",
    "is",
    "a",
    "guide",
    "to",
    "action",
    "which",
    "ensures",
    "that",
    "the",
    "military",
    "always",
    "obeys",
    "the",
    "commands",
    "of",
    "the",
    "party",
]

hypothesis2 = [
    "he",
    "read",
    "the",
    "book",
    "because",
    "he",
    "was",
    "interested",
    "in",
    "world",
    "history",
]


reference1 = [
    "It",
    "is",
    "a",
    "guide",
    "to",
    "action",
    "that",
    "ensures",
    "that",
    "the",
    "military",
    "will",
    "forever",
    "heed",
    "Party",
    "commands",
]


reference2 = [
    "It",
    "is",
    "the",
    "guiding",
    "principle",
    "which",
    "guarantees",
    "the",
    "military",
    "forces",
    "always",
    "being",
    "under",
    "the",
    "command",
    "of",
    "the",
    "Party",
]


reference3 = [
    "It",
    "is",
    "the",
    "practical",
    "guide",
    "for",
    "the",
    "army",
    "always",
    "to",
    "heed",
    "the",
    "directions",
    "of",
    "the",
    "party",
]

reference4 = [
    "he",
    "was",
    "interested",
    "in",
    "world",
    "history",
    "because",
    "he",
    "read",
    "the",
    "book",
]


references1 = [reference1, reference2, reference3]
references2 = [reference4]


@pytest.mark.parametrize(
    "sequence,n,result",
    [
        pytest.param(list(range(1, 4)), 1, [(1,), (2,), (3,)], id="unigrams"),
        pytest.param(list(range(1, 4)), 2, [(1, 2), (2, 3)], id="bigrams"),
        pytest.param("abc", 3, [("a", "b", "c")], id="trigrams"),
        pytest.param([1, 2, 3], 4, [], id="empty"),
    ],
)
def test_ngrams(
    sequence: Sequence[Hashable], n: int, result: Sequence[Hashable]
) -> None:
    assert list(ngrams(sequence, n)) == result


@pytest.mark.parametrize(
    "references,hypothesis,n,precision",
    [
        pytest.param(["abc"], "d", 1, Fraction(0, 1), id="zero score"),
        pytest.param(
            references1,
            ["of", "the"],
            1,
            Fraction(2, 2, _normalize=False),
            id="`of the` unigrams",
        ),
        pytest.param(
            references1,
            ["of", "the"],
            2,
            Fraction(1, 1),
            id="`of the` bigrams",
        ),
        pytest.param(
            references1,
            hypothesis1,
            1,
            Fraction(17, 18),
            id="normal unigrams",
        ),
        pytest.param(
            references1,
            hypothesis1,
            2,
            Fraction(10, 17),
            id="normal bigrams",
        ),
    ],
)
def test_modified_precision(
    references: Sequence[Sequence[Hashable]],
    hypothesis: Sequence[Hashable],
    n: int,
    precision: Fraction,
) -> None:
    assert modified_precision(references, hypothesis, n) == precision


@pytest.mark.parametrize(
    "references,hypothesis_length,closest_length",
    [
        pytest.param([list(range(1, 4))], 10, 3, id="[3,],10"),
        pytest.param(["a" * 20, "a" * 10, "a" * 8], 9, 8, id="[20, 10, 8],9"),
        pytest.param(["a" * 20, "a" * 10, "a" * 8], 17, 20, id="[20, 10, 8],17"),
    ],
)
def test_closest_reference_length(
    references: Sequence[Hashable], hypothesis_length: int, closest_length: int
) -> None:
    assert closest_reference_length(references, hypothesis_length) == closest_length


@pytest.mark.parametrize(
    "closest_reference_length,hypothesis_length,penalty",
    [(3, 3, 1.0), (3, 0, 0.0), (0, 3, 1.0), (3, 5, 1.0), (5, 3, 0.5134)],
)
def test_brevity_penalty(
    closest_reference_length: int, hypothesis_length: int, penalty
) -> None:
    assert (
        pytest.approx(
            brevity_penalty(closest_reference_length, hypothesis_length), rel=1e-4
        )
        == penalty
    )


@pytest.mark.parametrize(
    "precisions,weights,weighted_sum",
    [
        (
            (Fraction(10, 10, _normalize=False), Fraction(3, 6, _normalize=False)),
            (0.5, 0.5),
            0.75,
        ),
        (
            (Fraction(10, 10, _normalize=False), Fraction(0, 6, _normalize=False)),
            (0.5, 0.5),
            0.0,
        ),
    ],
)
def test_weighted_precision_sum(
    precisions: list[Fraction], weights: list[float], weighted_sum: float
) -> None:
    assert pytest.approx(weighted_precision_sum(precisions, weights) == weighted_sum)


@pytest.mark.parametrize(
    "list_of_references,hypotheses,weights,bleu",
    [
        pytest.param(
            [references1, references2],
            [hypothesis1, hypothesis2],
            (0.25, 0.25, 0.25, 0.25),
            0.5921,
            id="case1",
        ),
    ],
)
def test_corpus_bleu(
    list_of_references: Sequence[Sequence[Sequence[Hashable]]],
    hypotheses: Sequence[Sequence[Hashable]],
    weights: Sequence[float],
    bleu: float,
) -> None:
    assert (
        pytest.approx(
            corpus_bleu(list_of_references, hypotheses, weights=weights), rel=1e-4
        )
        == bleu
    )


@pytest.mark.parametrize(
    "references,hypothesis,weights,bleu",
    [
        pytest.param(
            references1,
            hypothesis1,
            (0.25, 0.25, 0.25, 0.25),
            0.5046,
            id="case1",
        ),
    ],
)
def test_sentence_bleu(
    references: Sequence[Sequence[Hashable]],
    hypothesis: Sequence[Hashable],
    weights: Sequence[float],
    bleu: float,
) -> None:
    assert (
        pytest.approx(sentence_bleu(references, hypothesis, weights=weights), rel=1e-4)
        == bleu
    )