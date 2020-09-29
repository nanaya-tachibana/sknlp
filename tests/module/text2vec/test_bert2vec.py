from collections import Counter

import numpy as np

from sknlp.vocab import Vocab
from sknlp.module.text2vec import Bert2vec

from .test_text2vec import TestText2vec


class TestBert2vec(TestText2vec):

    vocab = Vocab(Counter({"a": 10, "b": 12, "c": 22}))
    segmenter = None
    hidden_size = 768
    sequence_length = None
    max_sequence_length = 100
    num_layers = 3
    module = Bert2vec(
        vocab,
        segmenter,
        hidden_size=hidden_size,
        num_layers=num_layers,
        max_sequence_length=max_sequence_length,
        sequence_length=sequence_length
    )

    def test_save_load(self, tmp_path):
        filename = "tmp.tar"
        tmp_file = tmp_path / filename
        self.module.save_archive(str(tmp_file))
        new_module = Bert2vec.load_archive(str(tmp_file))
        np.testing.assert_array_almost_equal(
            self.module(np.array([["你好"]]))[0],
            new_module(np.array([["你好"]]))[0]
        )
        assert new_module.vocab["a"] == self.module.vocab["a"]

    def test_input_shape(self):
        assert self.module.input_shapes == [[-1, 1]]

    def test_output_shape(self):
        sequence_length = self.sequence_length or -1
        assert self.module.output_shapes == [
            [-1, sequence_length, self.hidden_size], [-1, self.hidden_size]
        ]

        sequence_length = 10
        module = Bert2vec(
            self.vocab,
            self.segmenter,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            max_sequence_length=self.max_sequence_length,
            sequence_length=sequence_length
        )
        assert module.output_shapes == [
            [-1, sequence_length, self.hidden_size], [-1, self.hidden_size]
        ]
        module._only_output_cls = True
        assert module.output_shapes == [[-1, self.hidden_size]]

    def test_get_config(self):
        super().test_get_config()
        config = self.module.get_config()
        assert config["hidden_size"] == self.hidden_size
