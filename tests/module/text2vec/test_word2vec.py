from collections import Counter

import numpy as np

from sknlp.vocab import Vocab
from sknlp.module.text2vec import Word2vec

from .test_text2vec import TestText2vec


class TestWord2vec(TestText2vec):

    vocab = Vocab(Counter({"a": 10, "b": 12, "c": 22}))
    embed_size = 10
    segmenter = "jieba"
    sequence_length = None
    max_sequence_length = 100
    module = Word2vec(vocab,
                      embed_size,
                      segmenter,
                      max_sequence_length=max_sequence_length,
                      sequence_length=sequence_length)

    def test_save_load(self, tmp_path):
        filename = "tmp.tar"
        tmp_file = tmp_path / filename
        self.module.save_archive(str(tmp_file))
        new_module = Word2vec.load_archive(str(tmp_file))
        np.testing.assert_array_almost_equal(
            self.module(np.array([[1, 3, 4]])), new_module(np.array([[1, 3, 4]]))
        )
        assert new_module.vocab["a"] == self.module.vocab["a"]

    def test_embedding_size(self):
        assert self.module.embedding_size == self.embed_size

    def test_get_config(self):
        super().test_get_config()
        config = self.module.get_config()
        assert config["embedding_size"] == self.embed_size
