import pytest
import numpy as np

from .bert_attention_mask import BertAttentionMaskLayer


@pytest.fixture
def type_ids() -> np.array:
    return np.array([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0]])


@pytest.fixture
def sequence_mask() -> np.array:
    return np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]])


def test_normal_bert_attention_mask(type_ids, sequence_mask) -> None:
    mask = np.array(
        [
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    np.testing.assert_array_almost_equal(
        BertAttentionMaskLayer()([type_ids, sequence_mask]), mask
    )


def test_unilm_bert_attention_mask(type_ids, sequence_mask) -> None:
    mask = np.array(
        [
            [
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    np.testing.assert_array_almost_equal(
        BertAttentionMaskLayer(mask_mode="unilm")([type_ids, sequence_mask]), mask
    )
