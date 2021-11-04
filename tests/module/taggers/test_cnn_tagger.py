import pytest
from sknlp.module.taggers import CNNTagger


@pytest.mark.parametrize(
    "output_format,use_raw_data",
    [
        pytest.param("bio", False, id="bio"),
        pytest.param("global_pointer", True, id="global_pointer"),
    ],
)
def test_cnn_tagger(
    output_format, use_raw_data, model_common_test, raw_data, file_data, word2vec
):
    labels = raw_data[0]
    model = CNNTagger(
        labels,
        output_format=output_format,
        crf_learning_rate_multiplier=20,
        dropout=0.0,
        text2vec=word2vec,
    )
    model_common_test(CNNTagger, model, raw_data[1:], file_data, use_raw_data, 5e-3)