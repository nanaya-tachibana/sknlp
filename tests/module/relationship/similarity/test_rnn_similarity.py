import pytest
from sknlp.module.relationship.similarity import RNNSimilarity


@pytest.mark.parametrize(
    "use_raw_data",
    [
        pytest.param(True, id="raw_data"),
        pytest.param(False, id="file_data"),
    ],
)
def test_rnn_similarity(use_raw_data, model_common_test, raw_data, file_data, word2vec):
    model = RNNSimilarity(dropout=0.2, text2vec=word2vec)
    model_common_test(RNNSimilarity, model, raw_data, file_data, use_raw_data, 1e-3)
