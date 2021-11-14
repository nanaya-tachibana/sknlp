import pytest
import numpy as np

from sknlp.utils.tagging import tagging_fscore, _compute_counts, Tag


@pytest.mark.parametrize(
    "label,prediction,classes,expected",
    [
        (
            [
                Tag(0, 1, "省"),
                Tag(2, 3, "市"),
                Tag(4, 6, "区"),
                Tag(7, 9, "详细"),
                Tag(10, 13, "详细"),
            ],
            [Tag(0, 1, "省"), Tag(2, 3, "市"), Tag(4, 6, "区"), Tag(10, 13, "详细")],
            ["省", "市", "区", "详细"],
            {
                "省": (1, 1, 1),
                "市": (1, 1, 1),
                "区": (1, 1, 1),
                "详细": (1, 2, 1),
                "avg": (4, 5, 4),
            },
        )
    ],
)
def test_compute_counts(label, prediction, classes, expected):
    counts = _compute_counts(label, prediction, classes)
    for count in counts:
        assert expected[count[0]] == count[1:]


@pytest.mark.parametrize(
    "labels,predictions,classes,expected",
    [
        (
            [
                [
                    Tag(0, 1, "省"),
                    Tag(2, 3, "市"),
                    Tag(4, 6, "区"),
                    Tag(7, 9, "详细"),
                    Tag(10, 13, "详细"),
                ],
                [],
                [Tag(0, 2, "市"), Tag(3, 5, "区")],
            ],
            [
                [Tag(0, 1, "省"), Tag(2, 3, "市"), Tag(4, 6, "区"), Tag(10, 13, "详细")],
                [Tag(0, 2, "省")],
                [Tag(3, 5, "区")],
            ],
            ["省", "市", "区", "详细"],
            {
                "省": (0.5, 1, 2 / 3),
                "市": (1, 0.5, 2 / 3),
                "区": (1, 1, 1),
                "详细": (1, 0.5, 2 / 3),
                "avg": (5 / 6, 5 / 7, 10 / 13),
            },
        ),
        (
            [[], [Tag(0, 2, "市"), Tag(3, 5, "区")]],
            [[Tag(0, 2, "省")], [Tag(3, 5, "区")]],
            ["省", "市", "区", "详细"],
            {
                "省": (0, 0, 0),
                "市": (0, 0, 0),
                "区": (1, 1, 1),
                "详细": (0, 0, 0),
                "avg": (0.5, 0.5, 0.5),
            },
        ),
    ],
)
def test_tagging_fscore(labels, predictions, classes, expected):
    score_df = tagging_fscore(labels, predictions, classes)
    classes.append("avg")
    for cls in classes:
        df = score_df[score_df["class"] == cls].iloc[0]
        np.testing.assert_array_almost_equal(
            expected[cls], (df.precision, df.recall, df.fscore)
        )
