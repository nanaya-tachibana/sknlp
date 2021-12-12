import json
import pytest
import tensorflow as tf
from sknlp.data.tagging_dataset import TaggingDataset, Vocab


@pytest.mark.parametrize(
    "max_length,output_format,add_start_end_tag, output",
    [
        pytest.param(
            100, "bio", False, [0, 1, 2, 0, 3, 4], id="bio without start and end"
        ),
        pytest.param(
            100, "bio", True, [0, 0, 1, 2, 0, 3, 4, 0], id="bio with start and end"
        ),
        pytest.param(2, "bio", True, [0, 0, 0, 0], id="truncated bio"),
        pytest.param(
            100,
            "global_pointer",
            False,
            [
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                ],
            ],
            id="global pointer without start and end",
        ),
        pytest.param(
            100,
            "global_pointer",
            True,
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ],
            id="global pointer with start and end",
        ),
        pytest.param(
            2,
            "global_pointer",
            True,
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            id="truncated global pointer",
        ),
    ],
)
def test_py_label_transform(max_length, output_format, add_start_end_tag, output):
    vocab = Vocab(["!", "@", "a", "bb", "111", "2"])
    classes = ["letter", "digit"]
    if output_format == "bio":
        classes.insert(0, "O")
    dataset = TaggingDataset(
        vocab,
        classes,
        X=["a", "a"],
        max_length=max_length,
        output_format=output_format,
        add_start_end_tag=add_start_end_tag,
    )
    tokens = ["!", "bb", "a", "@", "111", "2"]
    label = tf.constant(json.dumps([[1, 3, "letter"], [5, 8, "digit"]]))
    transformed_label = dataset.py_label_transform(label, tokens)
    assert transformed_label.tolist() == output
