import pytest
from .rcnn_classifier import RCNNClassifier


@pytest.mark.parametrize(
    "is_multilabel,use_raw_data",
    [
        pytest.param(True, True, id="multilabel"),
        pytest.param(False, False, id="singlelabel"),
    ],
)
def test_rnn_classifier(
    is_multilabel, use_raw_data, model_common_test, raw_data, file_data, word2vec
):
    labels = raw_data[0]
    model = RCNNClassifier(
        labels, is_multilabel=is_multilabel, dropout=0.2, text2vec=word2vec
    )
    model_common_test(
        RCNNClassifier, model, raw_data[1:], file_data, use_raw_data, 5e-3
    )