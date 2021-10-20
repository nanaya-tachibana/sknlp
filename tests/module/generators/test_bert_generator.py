import pytest
from sknlp.module.generators import BertGenerator


@pytest.mark.parametrize(
    "use_raw_data",
    [
        pytest.param(True, id="raw_data"),
        pytest.param(False, id="file_data"),
    ],
)
def test_bert_generator(use_raw_data, model_common_test, raw_data, file_data, bert2vec):
    model = BertGenerator(dropout=0.0, text2vec=bert2vec)
    model_common_test(BertGenerator, model, raw_data, file_data, use_raw_data, 1e-4)