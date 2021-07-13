from __future__ import annotations
from typing import Sequence, Callable
from dataclasses import dataclass
import itertools

import pandas as pd


@dataclass
class Tag:
    start: int
    end: str
    label: str

    def __hash__(self) -> int:
        return hash((self.label, self.start, self.end))


def convert_ids_to_tags(
    tag_ids: Sequence[int],
    idx2class: Callable[[int], str],
    add_start_end_tag: bool = False,
) -> list[Tag]:
    if add_start_end_tag:
        tag_ids = tag_ids[1:]
    current_begin_tag = -1
    begin = 0
    parsed_tags = list()
    for i, tag_id in enumerate(tag_ids):
        if tag_id != 0 and tag_id % 2 == 0 and tag_id - 1 == current_begin_tag:
            continue

        if i != begin:
            parsed_tags.append(
                Tag(begin, i - 1, idx2class((current_begin_tag + 1) // 2))
            )

        if tag_id % 2 == 1:
            begin = i
            current_begin_tag = tag_id
        else:
            begin = i + 1
            current_begin_tag = -1
    if begin != len(tag_ids) and current_begin_tag != -1:
        parsed_tags.append(
            Tag(begin, len(tag_ids) - 1, idx2class((current_begin_tag + 1) // 2))
        )
    return parsed_tags


def _compute_counts(
    label: Sequence[Tag], prediction: Sequence[Tag], classes: Sequence[str]
) -> list[tuple[str, int, int, int]]:
    truth_tags = set(label)
    prediction_tags = set(prediction)
    correct_tags = truth_tags & prediction_tags
    counts = [
        (
            cls,
            len([i for i, tag in enumerate(correct_tags) if tag.label == cls]),
            len([i for i, tag in enumerate(truth_tags) if tag.label == cls]),
            len([i for i, tag in enumerate(prediction_tags) if tag.label == cls]),
        )
        for cls in classes
    ]
    counts.append(("avg", len(correct_tags), len(truth_tags), len(prediction_tags)))
    return counts


def tagging_fscore(
    y: Sequence[Sequence[Tag]],
    p: Sequence[Sequence[Tag]],
    classes: Sequence[str],
) -> pd.DataFrame:
    df = pd.DataFrame(
        itertools.chain.from_iterable(
            _compute_counts(yi, pi, classes) for yi, pi in zip(y, p)
        ),
        columns=["class", "correct", "support", "prediction"],
    )
    df = df.groupby("class").sum()
    df["precision"] = df.correct / df.prediction
    df["recall"] = df.correct / df.support
    df["fscore"] = 2 * df.precision * df.recall / (df.precision + df.recall)
    df.fillna(0, inplace=True)
    df.reset_index(inplace=True)
    return df[["class", "precision", "recall", "fscore", "support"]]