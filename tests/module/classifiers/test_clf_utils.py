import pytest

import numpy as np

from sknlp.module.classifiers.utils import (
    _validate_thresholds,
    logits2probabilities,
    probabilities2classes,
    logits2classes,
    precision_recall_fscore,
    label_binarizer,
    classification_fscore
)


@pytest.mark.parametrize(
    "logits,is_multilabel,expected",
    [
        (np.array([[0, 10]]), True, np.array([[0.5, 1]])),
        (np.array([[0, 10]]), False, np.array([[0, 1]])),
        (np.array([0, 10]), False, None)
    ],
    ids=["multilabel", "single label", "invalid input"]
)
def test_logits2probabilities(logits, is_multilabel, expected):
    if expected is None:
        with pytest.raises(ValueError):
            logits2probabilities(logits, is_multilabel)
    else:
        np.testing.assert_almost_equal(
            logits2probabilities(logits, is_multilabel), expected,
            decimal=4
        )


@pytest.mark.parametrize(
    "thresholds,length,expected",
    [
        ([0.5, 0.1], 2, [0.5, 0.1]),
        (0.5, 2, [0.5, 0.5]),
        ([0.5, 0.7, 0.2], 2, None)
    ],
    ids=["list input", "float input", "invalid length"]
)
def test_validate_thresholds(thresholds, length, expected):
    if expected is None:
        with pytest.raises(ValueError):
            _validate_thresholds(thresholds, length)
    else:
        assert _validate_thresholds(thresholds, length) == expected


@pytest.mark.parametrize(
    "probabilities,is_multilabel,thresholds,expected",
    [
        (np.array([[0.4, 0.8]]), True, 0.5, [[1]]),
        (np.array([[0.4, 0.8]]), True, [0.5, 0.9], [[]]),
        (np.array([[0.4, 0.6], [0.2, 0.8]]), False, 0.6, [0, 1]),
        (np.array([0.4, 0.6]), False, 0.8, None)
    ],
    ids=[
        "multilabel with same threshold",
        "multilabel with different thresholds",
        "single label",
        "invalid input"
    ]
)
def test_probabilities2classes(probabilities, is_multilabel, thresholds, expected):
    if expected is None:
        with pytest.raises(ValueError):
            probabilities2classes(probabilities, is_multilabel, thresholds)
    else:
        assert (
            probabilities2classes(probabilities, is_multilabel, thresholds) == expected
        )


@pytest.mark.parametrize(
    "logits,is_multilabel,thresholds,expected",
    [
        (np.array([[0, 10]]), True, 0.5, [[1]]),
        (np.array([[0, 1]]), True, [0.5, 0.9], [[]]),
        (np.array([[0, 10], [1, 5]]), False, 0.8, [1, 1]),
        (np.array([0, 8]), False, 0.8, None)
    ],
    ids=[
        "multilabel with same threshold",
        "multilabel with different thresholds",
        "single label",
        "invalid input"
    ]
)
def test_logits2classes(logits, is_multilabel, thresholds, expected):
    if expected is None:
        with pytest.raises(ValueError):
            logits2classes(logits, is_multilabel, thresholds)
    else:
        assert logits2classes(logits, is_multilabel, thresholds) == expected


@pytest.mark.parametrize(
    "tp,fp,fn,beta,precision,recall,fscore",
    [
        (1, 1, 3, 1, 0.5, 0.25, 0.33333333),
        (1, 1, 3, 2, 0.5, 0.25, 0.27777778),
        (1, 1, 3, 0.5, 0.5, 0.25, 0.41666667),
        (0, 1, 0, 1, 0, 0, 0),
        (0, 0, 1, 1, 0, 0, 0)
    ],
    ids=[
        "normal input beta=1",
        "normal input beta=2",
        "normal input beta=0.5",
        "tp = fn = 0",
        "tp = fp = 0"
    ]
)
def test_precision_recall_fscore(tp, fp, fn, beta, precision, recall, fscore):
    p, r, f = precision_recall_fscore(tp, fp, fn, beta)
    assert p == precision
    assert r == recall
    np.testing.assert_almost_equal(f, fscore)


@pytest.mark.parametrize(
    "y,p,classes,y_one_hot,p_one_hot,return_classes",
    [
        (
            ["a", "a", "b", "c"], ["a", "b", "c", "c"], ["a", "b", "c"],
            np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]),
            ["a", "b", "c"]
        ),
        (
            ["a", "a", "b", "c"], ["a", "b", "c", "c"], None,
            np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]),
            ["a", "b", "c"]
        ),
        (
            ["a", "a", "b", "c"], ["a", "b", "c", "c"], ["a", "b"],
            np.array([[1, 0], [1, 0], [0, 1], [0, 0]]),
            np.array([[1, 0], [0, 1], [0, 0], [0, 0]]),
            ["a", "b"]
        ),
        (
            [["a"], ["a", "b"], ["c"]], [["a", "b"], ["b"], ["c"]], None,
            np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]]),
            np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
            ["a", "b", "c"]
        )
    ],
    ids=[
        "single label with classes",
        "single label without classes",
        "single label with less classes",
        "multilabel",
    ]
)
def test_label_binarizer(y, p, classes, y_one_hot, p_one_hot, return_classes):
    y_, p_, classes_ = label_binarizer(y, p, classes)
    np.testing.assert_equal(y_, y_one_hot)
    np.testing.assert_equal(p_, p_one_hot)
    assert classes_ == return_classes


@pytest.mark.parametrize(
    "y,p,is_multilabel,expected_shape",
    [
        (["a", "a", "b", "c"], ["a", "b", "c", "c"], False, (4, 9)),
        ([["a"], ["a", "b"], ["c"]], [["a", "b"], ["b"], ["c"]], True, (4, 9))
    ],
    ids=[
        "single label",
        "multilabel",
    ]
)
def test_classification_fscore(y, p, is_multilabel, expected_shape):
    df = classification_fscore(y, p, is_multilabel)
    assert df.shape == expected_shape
    if not is_multilabel:
        row = df[df["class"] == "avg"]
        assert row.precision.values == row.recall.values == row.fscore.values
