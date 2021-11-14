import pytest
from sknlp.module.retrievers import BertRetriever


@pytest.mark.parametrize(
    "use_raw_data",
    [
        pytest.param(True, id="raw_data"),
        pytest.param(False, id="file_data"),
    ],
)
def test_bert_retriever(use_raw_data, model_common_test, raw_data, file_data, bert2vec):
    model = BertRetriever(dropout=0.1, has_negative=True, text2vec=bert2vec)
    model_common_test(BertRetriever, model, raw_data, file_data, use_raw_data, 2e-4)