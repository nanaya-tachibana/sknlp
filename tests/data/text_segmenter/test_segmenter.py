import pytest

from sknlp.data.text_segmenter import get_segmenter


def test_get_segmenter():
    assert get_segmenter('list')('xyz') == ['x', 'y', 'z']
    with pytest.raises(ValueError):
        assert get_segmenter('hanlp')
