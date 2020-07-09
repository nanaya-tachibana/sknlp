import numpy as np

from sknlp.module.classifiers import TextRNNClassifier
from .test_clf_deep_classifier import TestDeepClassifier


class TestTextRNNClassifier(TestDeepClassifier):

    name = "pp"
    classes = ["1", "2", "3"]
    word2vec = None
    segmenter = "char"
    max_sequence_length = 90
    model = TextRNNClassifier(
        classes,
        max_sequence_length=max_sequence_length,
        segmenter=segmenter,
        text2vec=word2vec,
        is_multilabel=False,
        name=name
    )
    model.fit(
        X=["111", "222", "aaa", "bbb", "a4b", "111", "222", "aaa", "bbb", "a4b"],
        y=["1", "1", "2", "2", "3", "1", "1", "2", "2", "3"],
        valid_X=["111", "222", "aaa", "bbb", "a4b"],
        valid_y=["1", "1", "2", "2", "3"],
        n_epochs=1, batch_size=2
    )

    def test_save_load(self, tmp_path):
        self.model.save(str(tmp_path))
        new_model = TextRNNClassifier.load(str(tmp_path))
        np.testing.assert_array_almost_equal(
            self.model._model.predict(np.array([[1, 3, 4]])),
            new_model._model.predict(np.array([[1, 3, 4]]))
        )
