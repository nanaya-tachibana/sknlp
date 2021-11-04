from __future__ import annotations
from typing import Sequence, Callable, Union
from dataclasses import dataclass
import itertools

import pandas as pd
import numpy as np
from scipy.special import expit

from .classification import _validate_thresholds


@dataclass
class Tag:
    start: int
    end: str
    label: str

    def __hash__(self) -> int:
        return hash((self.label, self.start, self.end))


class IdentityDict(dict):
    def __getitem__(self, key):
        return key


def convert_ids_to_tags(
    tag_ids: Sequence[int],
    idx2class: Callable[[int], str],
    itoken2ichar: tuple[dict[int, int], dict[int, int]] | None = None,
    add_start_end_tag: bool = False,
) -> list[Tag]:
    if add_start_end_tag:
        tag_ids = tag_ids[1:-1]
    if itoken2ichar is None:
        start_mapping = IdentityDict()
        end_mapping = IdentityDict()
    else:
        start_mapping, end_mapping = itoken2ichar
    num_tag_ids = len(tag_ids)
    current_begin_tag = -1
    begin = 0
    parsed_tags = list()
    for i, tag_id in enumerate(tag_ids):
        if (
            i < num_tag_ids - 1
            and tag_id != 0
            and tag_id % 2 == 0
            and tag_id - 1 == current_begin_tag
        ):
            continue

        if i != begin:
            parsed_tags.append(
                Tag(
                    start_mapping[begin],
                    end_mapping[i - (i < num_tag_ids - 1)],
                    idx2class((current_begin_tag + 1) // 2),
                )
            )

        if tag_id % 2 == 1:
            begin = i
            current_begin_tag = tag_id
        else:
            begin = i + 1
            current_begin_tag = -1
    return parsed_tags


def convert_global_pointer_to_tags(
    pointer: np.ndarray,
    thresholds: Union[float, list[float]],
    idx2class: Callable[[int], str],
    itoken2ichar: tuple[dict[int, int], dict[int, int]] | None = None,
    add_start_end_tag: bool = False,
) -> list[Tag]:
    thresholds = _validate_thresholds(thresholds, pointer.shape[0])
    if itoken2ichar is None:
        start_mapping = IdentityDict()
        end_mapping = IdentityDict()
    else:
        start_mapping, end_mapping = itoken2ichar
    tags: list[Tag] = []
    for i, score_matrix in enumerate(pointer):
        label = idx2class(i)
        for start, end in zip(*np.where(expit(score_matrix) >= thresholds[i])):
            start -= add_start_end_tag
            end -= add_start_end_tag
            tags.append(Tag(start_mapping[int(start)], end_mapping[int(end)], label))
    return tags


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
    df["TP"] = df.correct
    df["FP"] = df.prediction - df.correct
    df["FN"] = df.support - df.correct
    df.fillna(0, inplace=True)
    df.reset_index(inplace=True)
    df["order"] = range(df.shape[0])
    df.loc[df["class"] == "avg", "order"] = df.shape[0] + 1
    df = df.sort_values("order").drop("order", axis=1)
    return df[["class", "precision", "recall", "fscore", "support", "TP", "FP", "FN"]]