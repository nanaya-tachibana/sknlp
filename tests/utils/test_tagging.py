import pytest
import numpy as np

from sknlp.utils.tagging import tagging_fscore, _compute_counts


@pytest.mark.parametrize(
    "text,label,prediction,classes,expected",
    [
        (
            "江苏苏州吴中区和平路同心家园",
            [
                "B-省",
                "I-省",
                "B-市",
                "I-市",
                "B-区",
                "I-区",
                "I-区",
                "B-详细",
                "I-详细",
                "I-详细",
                "B-详细",
                "I-详细",
                "I-详细",
                "I-详细",
            ],
            [
                "B-省",
                "I-省",
                "B-市",
                "I-市",
                "B-区",
                "I-区",
                "I-区",
                "O",
                "O",
                "O",
                "B-详细",
                "I-详细",
                "I-详细",
                "I-详细",
            ],
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
def test_compute_counts(text, label, prediction, classes, expected):
    counts = _compute_counts(text, label, prediction, classes)
    for count in counts:
        assert expected[count[0]] == count[1:]


@pytest.mark.parametrize(
    "texts,labels,predictions,classes,expected",
    [
        (
            ["江苏苏州吴中区和平路同心家园", "你我他", "北京市通州区"],
            [
                [
                    "B-省",
                    "I-省",
                    "B-市",
                    "I-市",
                    "B-区",
                    "I-区",
                    "I-区",
                    "B-详细",
                    "I-详细",
                    "I-详细",
                    "B-详细",
                    "I-详细",
                    "I-详细",
                    "I-详细",
                ],
                ["O", "O", "O"],
                ["B-市", "I-市", "I-市", "B-区", "I-区", "I-区"],
            ],
            [
                [
                    "B-省",
                    "I-省",
                    "B-市",
                    "I-市",
                    "B-区",
                    "I-区",
                    "I-区",
                    "O",
                    "O",
                    "O",
                    "B-详细",
                    "I-详细",
                    "I-详细",
                    "I-详细",
                ],
                ["B-省", "I-省", "I-省"],
                ["O", "O", "O", "B-区", "I-区", "I-区"],
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
            ["你我他", "北京市通州区"],
            [
                ["O", "O", "O"],
                ["B-市", "I-市", "I-市", "B-区", "I-区", "I-区"],
            ],
            [
                ["B-省", "I-省", "I-省"],
                ["O", "O", "O", "B-区", "I-区", "I-区"],
            ],
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
def test_tagging_fscore(texts, labels, predictions, classes, expected):
    score_df = tagging_fscore(texts, labels, predictions, classes)
    classes.append("avg")
    for cls in classes:
        df = score_df[score_df["class"] == cls].iloc[0]
        np.testing.assert_array_almost_equal(
            expected[cls], (df.precision, df.recall, df.fscore)
        )
