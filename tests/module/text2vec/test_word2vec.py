from collections import Counter

import numpy as np

from sknlp.vocab import Vocab
from sknlp.module.text2vec import Word2vec

from .test_text2vec import TestText2vec


class TestToken2vec(TestText2vec):

    vocab = Vocab(Counter({'a': 10, 'b': 12, 'c': 22}))
    embed_size = 10
    segmenter = 'jieba'
    module = Word2vec(vocab, embed_size, segmenter)

    def test_save_load(self, tmp_path):
        filename = 'tmp.tar'
        tmp_file = tmp_path / filename
        self.module.save_archive(str(tmp_file))
        new_module = Word2vec.load_archive(str(tmp_file))
        np.testing.assert_array_almost_equal(
            self.module(np.array([[1, 3, 4]])), new_module(np.array([[1, 3, 4]]))
        )
        assert new_module.vocab["a"] == self.module.vocab["a"]

    def test_input_shape(self):
        assert self.module.input_shape.as_list() == [None, None]

    def test_output_shape(self):
        assert self.module.output_shape.as_list() == [None, None, self.embed_size]

    def test_get_config(self):
        super().test_get_config()
        config = self.module.get_config()
        assert config["embedding_size"] == self.embed_size
