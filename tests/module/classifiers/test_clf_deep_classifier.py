from sknlp.module.classifiers.deep_classifier import DeepClassifier
from ..test_supervised_model import TestSupervisedNLPModel


class TestDeepClassifier(TestSupervisedNLPModel):

    name = "pp"
    word2vec = None
    classes = ['1', '2', '3']
    word2vec = None
    segmenter = "jieba"
    max_length = 100
    model = DeepClassifier(
        classes,
        max_length=max_length,
        segmenter=segmenter,
        text2vec=word2vec,
        is_multilabel=False,
        name=name
    )

    def test_get_config(self):
        super().test_get_config()
        assert self.model.get_config()['is_multilabel'] is False
