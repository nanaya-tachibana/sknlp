from collections import Counter

import numpy as np

from sknlp.vocab import Vocab
from sknlp.module.embedding import Token2vec


class TestToken2vec:

    vocab = Vocab(Counter({'a': 10, 'b': 12, 'c': 22}))
    embed_size = 10
    segmenter = 'jieba'
    module = Token2vec(vocab, embed_size, segmenter,
                       embeddings_initializer='uniform')

    def test_vocab(self):
        assert self.module.vocab['a'] == self.vocab['a']

    def test_segmeter(self):
        assert self.module.segmenter == self.segmenter

    def test_embed_size(self):
        assert self.module.embed_size == self.embed_size

    def test_save_load(self, tmp_path):
        tmp_file = tmp_path / 'tmp.tar'
        self.module.save(str(tmp_file))
        new_module = Token2vec.load(str(tmp_file))
        np.testing.assert_array_almost_equal(self.module(np.array([1, 3, 5])),
                                             new_module(np.array([1, 3, 5])))

    def test_input_shape(self):
        assert self.module.input_shape.as_list() == [None, None]

    def test_output_shape(self):
        assert self.module.output_shape.as_list() == [None, None,
                                                      self.embed_size]
