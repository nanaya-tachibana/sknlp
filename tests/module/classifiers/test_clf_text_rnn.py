import numpy as np

from sknlp.module.classifiers import TextRNNClassifier
from tests.module.classifiers.test_clf_deep_classifier import \
    TestDeepClassifier


class TestTextRNNClassifier(TestDeepClassifier):

    model = TextRNNClassifier(
        ["1", "2", "3"],
        vocab=TextRNNClassifier.build_vocab(
            ["11111", "22222", "44444", "5555", "aaaa", "bbbb", "cccc"], list
        ),
        is_multilabel=False
    )

    def test_save_load(self, tmp_path):
        self.model.build()
        self.model.fit(
            X=["111", "222", "aaa", "bbb", "a4b"],
            y=["1", "1", "2", "2", "3"],
            valid_X=["111", "222", "aaa", "bbb", "a4b"],
            valid_y=["1", "1", "2", "2", "3"],
            n_epochs=1, batch_size=2
        )
        self.model.save(str(tmp_path))
        new_model = TextRNNClassifier.load(str(tmp_path))
        np.testing.assert_array_almost_equal(
            self.model._model.predict(np.array([[1, 3, 5]])),
            new_model._model.predict(np.array([[1, 3, 5]]))
        )
