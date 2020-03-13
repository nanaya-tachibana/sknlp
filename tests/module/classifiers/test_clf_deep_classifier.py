from sknlp.module.classifiers.deep_classifier import DeepClassifier
from tests.module.test_base_model import TestSupervisedNLPModel


class TestDeepClassifier(TestSupervisedNLPModel):

    model = DeepClassifier(['1', '2', '3'], is_multilabel=False)

    def test_config(self):
        super().test_config()
        assert self.model.get_config()['is_multilabel'] is False
