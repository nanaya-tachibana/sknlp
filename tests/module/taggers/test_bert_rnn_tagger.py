import pytest
from sknlp.module.taggers import BertRNNTagger


@pytest.mark.parametrize(
    "output_format,use_raw_data",
    [
        pytest.param("bio", False, id="bio"),
        pytest.param("global_pointer", True, id="global_pointer"),
    ],
)
def test_bert_rnn_tagger(
    output_format, use_raw_data, model_common_test, raw_data, file_data, bert2vec
):
    labels = raw_data[0]
    model = BertRNNTagger(
        labels,
        output_format=output_format,
        crf_learning_rate_multiplier=2000,
        rnn_learning_rate_multiplier=100,
        dropout=0.1,
        text2vec=bert2vec,
    )
    model_common_test(BertRNNTagger, model, raw_data[1:], file_data, use_raw_data, 5e-5)