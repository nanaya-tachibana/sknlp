from typing import Tuple, List, Sequence
from dataclasses import dataclass
import itertools

import numpy as np
import pandas as pd


@dataclass
class Tag:
    label: str
    start: int
    end: str
    text: str

    def __hash__(self):
        return hash((self.label, self.start, self.end, self.text))


def parse_tagged_text(text: str, tag_names: Sequence[str]) -> List[Tag]:
    current_label = None
    start = 0
    parsed_tags = list()
    for i, tag_name in enumerate(tag_names):
        if tag_name == "O":
            tag_type, tag_label = "O", None
        else:
            tag_type, tag_label = tag_name.split("-")
        if (tag_type == "I" or tag_type == "E") and tag_label == current_label:
            continue

        if i != start:
            parsed_tags.append(Tag(current_label, start, i, text[start:i]))

        if tag_type == "B" or tag_type == "S":
            start = i
            current_label = tag_label
        else:
            start = i + 1
            current_label = None
    if start != len(tag_names) and current_label is not None:
        parsed_tags.append(Tag(current_label, start, len(tag_names), text[start:]))
    return parsed_tags


def _compute_counts(
    text: str, label: Sequence[str], prediction: Sequence[str], classes: Sequence[str]
) -> List[Tuple[str, int, int, int]]:
    truth_tags = set(parse_tagged_text(text, label))
    prediction_tags = set(parse_tagged_text(text, prediction))
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
    texts: Sequence[str],
    y: Sequence[Sequence[str]],
    p: Sequence[Sequence[str]],
    classes: Sequence[str],
) -> pd.DataFrame:
    df = pd.DataFrame(
        itertools.chain.from_iterable(
            _compute_counts(text, yi, pi, classes) for text, yi, pi in zip(texts, y, p)
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
