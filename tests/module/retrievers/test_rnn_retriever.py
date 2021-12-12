import pytest
from sknlp.module.retrievers import RNNRetriever


@pytest.mark.parametrize(
    "use_raw_data",
    [
        pytest.param(True, id="raw_data"),
        pytest.param(False, id="file_data"),
    ],
)
def test_rnn_retriever(use_raw_data, model_common_test, raw_data, file_data, word2vec):
    model = RNNRetriever(rnn_dropout=0.2, has_negative=True, text2vec=word2vec)
    model_common_test(RNNRetriever, model, raw_data, file_data, use_raw_data, 1e-3)