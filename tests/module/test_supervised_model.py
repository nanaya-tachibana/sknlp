from sknlp.module.supervised_model import SupervisedNLPModel
from sknlp.module.text2vec import Word2vec
from sknlp.vocab import Vocab

from .test_base_model import TestBaseNLPModel


class TestSupervisedNLPModel(TestBaseNLPModel):

    vocab = Vocab()
    embedding_size = 100
    word2vec = Word2vec(vocab, embedding_size, segmenter="char")
    classes = ["a", "b", "c"]
    name = "yy"
    segmenter = "jieba"
    max_sequence_length = 100
    model = SupervisedNLPModel(
        classes,
        max_sequence_length=max_sequence_length,
        segmenter=segmenter,
        text2vec=word2vec,
        name=name
    )

    def test_init(self):
        if self.word2vec is not None:
            assert self.model._segmenter == self.word2vec.segmenter
            assert self.model._embedding_size == self.word2vec.embedding_size
        else:
            assert self.model._segmenter == self.segmenter

    def test_get_config(self):
        super().test_get_config()
        config = self.model.get_config()
        assert config["classes"] == self.classes
