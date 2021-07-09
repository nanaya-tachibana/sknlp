from __future__ import annotations
from typing import Optional, Sequence, Callable
from dataclasses import dataclass
import itertools

import numpy as np
import pandas as pd


@dataclass
class Tag:
    start: int
    end: str
    label: str

    def __hash__(self) -> int:
        return hash((self.label, self.start, self.end))


def parse_tags(tag_names: Sequence[str]) -> list[Tag]:
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
            parsed_tags.append(Tag(start, i - 1, current_label))

        if tag_type == "B" or tag_type == "S":
            start = i
            current_label = tag_label
        else:
            start = i + 1
            current_label = None
    if start != len(tag_names) and current_label is not None:
        parsed_tags.append(Tag(start, len(tag_names) - 1, current_label))
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


def convert_ids_to_tags(
    idx2tag: Callable[[int], str],
    tag_ids: list[int],
    pad_tag: str,
    start_tag: Optional[str] = None,
    end_tag: Optional[str] = None,
) -> list[Tag]:
    exclude_tags = {pad_tag, start_tag, end_tag}
    tags = []
    for tag_id in tag_ids:
        tag = idx2tag(tag_id)
        if tag not in exclude_tags:
            tags.append(tag)
        if tag == pad_tag:
            break
    return parse_tags(tags)


def viterbi_decode(transitions, emissions, mask=None):
    """
    Decode the highest scoring sequence of tags outside of tensorflow.
    Parameters:
    ----
    transitions: numpy array, shape(num_tags, num_tags)
    emissions: numpy array, shape(seq_length, batch_size, num_tags)
    mask: numpy array, shape(seq_length, batch_size)
    Returns:
    ----
    best_tags_list: list of list
      Each list in best_tags_list is the best path of each sequence.
    """
    seq_legnth, batch_size, num_tags = emissions.shape
    if mask is None:
        mask = np.ones((seq_legnth, batch_size))
    assert mask[0].all()

    sequence_lengths = mask.sum(axis=0).astype(np.int64)
    # list to store the decode paths
    best_tags_list = []
    # (seq_length, batch_size, num_tags)
    viterbi_scores = np.zeros_like(emissions, dtype=np.float32)
    viterbi_scores[0] = emissions[0]
    # (seq_length, batch_size, num_tags)
    viterbi_paths = np.zeros_like(emissions, dtype=np.int64)

    # use dynamic programing to compute the viterbi score
    for i, emission in enumerate(emissions[1:]):
        # (batch_size, num_tags, 1)
        broadcast_log_prob = viterbi_scores[i].reshape((-1, num_tags, 1))
        # (batch_size, num_tags, num_tags)
        score = broadcast_log_prob + transitions
        viterbi_paths[i] = score.argmax(axis=1)
        viterbi_scores[i + 1] = emission + score.max(axis=1)
    # search best path for each batch according to the viterbi score
    # get the best tag for the last emission
    for idx in range(batch_size):
        seq_end_idx = sequence_lengths[idx] - 1
        best_last_tag = viterbi_scores[seq_end_idx, idx].argmax()
        best_tags = [best_last_tag]
        # trace back all best tags based on the last best tag and viterbi path
        for path in np.flip(viterbi_paths[: sequence_lengths[idx] - 1], 0):
            best_last_tag = path[idx][best_tags[-1]]
            best_tags.append(best_last_tag)
        best_tags.reverse()
        best_tags_list.append(best_tags)
    return best_tags_list