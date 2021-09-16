# Natural Language Toolkit: BLEU Score
#
# Copyright (C) 2001-2021 NLTK Project
# Authors: Chin Yee Lee, Hengfeng Li, Ruxin Hou, Calvin Tanujaya Lim
# Contributors: Björn Mattsson, Dmitrijs Milajevs, Liling Tan
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
# 实现参考https://github.com/nltk/nltk/blob/3.6.2/nltk/translate/bleu_score.py

from __future__ import annotations
from typing import Iterable, Optional, Sequence, Hashable, Callable

import math
import warnings
from collections import Counter
import functools
from fractions import Fraction


def ngrams(sequence: Sequence[Hashable], n: int) -> Iterable:
    return zip(*[sequence[i:] for i in range(n)])


def modified_precision(
    references: Sequence[Sequence[Hashable]], hypothesis: Sequence[Hashable], n: int
) -> Fraction:
    ngram_counts = Counter(ngrams(hypothesis, n))
    max_counts = functools.reduce(
        Counter.__or__, (Counter(ngrams(ref, n)) for ref in references)
    )
    clipped_ngram_counts = ngram_counts & max_counts
    numerator = sum(clipped_ngram_counts.values())
    denominator = max(1, sum(ngram_counts.values()))
    return Fraction(numerator, denominator, _normalize=False)


def closest_reference_length(
    referecnes: Sequence[Hashable], hypothesis_length: int
) -> int:
    if len(referecnes) == 1:
        return len(referecnes[0])
    return min(
        (len(ref) for ref in referecnes),
        key=lambda ref_len: (abs(ref_len - hypothesis_length), ref_len),
    )


def brevity_penalty(closest_reference_length: int, hypothesis_length: int) -> int:
    if hypothesis_length == 0:
        return 0
    elif hypothesis_length >= closest_reference_length:
        return 1
    else:
        return math.exp(1 - closest_reference_length / hypothesis_length)


def weighted_precision_sum(precisions: list[Fraction], weights: list[float]) -> float:
    weighted_log_p = []
    for i, (weight, precision) in enumerate(zip(weights, precisions), start=1):
        if precision.numerator == 0.0:
            log_p = float("-inf")
            warnings.warn(f"{i}th order ngram的precision为0.")
        else:
            log_p = math.log(precision)
        weighted_log_p.append(weight * log_p)
    return math.exp(math.fsum(weighted_log_p))


def corpus_bleu(
    list_of_references: Sequence[Sequence[Sequence[Hashable]]],
    hypotheses: Sequence[Sequence[Hashable]],
    weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    combine_ngram_precision: Optional[
        Callable[[list[float], list[Fraction]], float]
    ] = None,
) -> float:
    if len(list_of_references) != len(hypotheses):
        warnings.warn(
            f"`list_of_references`和`hypotheses`长度不一致, "
            f"长度为{len(list_of_references)}和{len(hypotheses)}."
        )
    max_order = len(weights)
    p_numerator: list[int] = [0 for _ in range(max_order)]
    p_denominator: list[int] = [0 for _ in range(max_order)]

    reference_total_length = 0
    hypothesis_total_length = 0
    for references, hypothesis in zip(list_of_references, hypotheses):
        for i in range(max_order):
            p_i = modified_precision(references, hypothesis, i + 1)
            p_numerator[i] += p_i.numerator
            p_denominator[i] += p_i.denominator
        hypothesis_length = len(hypothesis)
        hypothesis_total_length += hypothesis_length
        reference_total_length += closest_reference_length(
            references, hypothesis_length
        )

    p_n: list[Fraction] = [Fraction(n, d) for n, d in zip(p_numerator, p_denominator)]
    if p_n[0].numerator == 0:
        return 0

    if combine_ngram_precision is None:
        combine_ngram_precision = weighted_precision_sum
    score = weighted_precision_sum(p_n, weights)
    bp = brevity_penalty(reference_total_length, hypothesis_total_length)
    return bp * score


def sentence_bleu(
    references: Sequence[Sequence[Hashable]],
    hypothesis: Sequence[Hashable],
    weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    combine_ngram_precision: Optional[
        Callable[[list[float], list[Fraction]], float]
    ] = None,
) -> float:
    return corpus_bleu(
        [references],
        [hypothesis],
        weights=weights,
        combine_ngram_precision=combine_ngram_precision,
    )
