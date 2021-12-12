import pytest
from sknlp.module.classifiers import BertClassifier


def test_bert_classifier_recompute(
    model_common_test, raw_data, file_data, recompute_bert2vec
):
    labels = raw_data[0]
    model = BertClassifier(
        labels, is_multilabel=False, cls_dropout=0.1, text2vec=recompute_bert2vec
    )
    model_common_test(BertClassifier, model, raw_data[1:], file_data, False, 1e-4)


@pytest.mark.parametrize(
    "is_multilabel,use_raw_data",
    [
        pytest.param(True, True, id="multilabel"),
        pytest.param(False, False, id="singlelabel"),
    ],
)
def test_bert_classifier(
    is_multilabel, use_raw_data, model_common_test, raw_data, file_data, bert2vec
):
    labels = raw_data[0]
    model = BertClassifier(
        labels, is_multilabel=is_multilabel, cls_dropout=0.1, text2vec=bert2vec
    )
    model_common_test(
        BertClassifier, model, raw_data[1:], file_data, use_raw_data, 1e-4
    )


@pytest.mark.parametrize(
    "is_multilabel,use_raw_data",
    [
        pytest.param(True, True, id="multilabel"),
        pytest.param(False, False, id="singlelabel"),
    ],
)
def test_pairwise_bert_classifier(
    is_multilabel,
    use_raw_data,
    model_common_test,
    raw_data_pairwise,
    file_data_pairwise,
    bert2vec,
):
    labels = raw_data_pairwise[0]
    model = BertClassifier(
        labels,
        is_pair_text=True,
        is_multilabel=is_multilabel,
        cls_dropout=0.1,
        text2vec=bert2vec,
    )
    model_common_test(
        BertClassifier,
        model,
        raw_data_pairwise[1:],
        file_data_pairwise,
        use_raw_data,
        1e-4,
    )