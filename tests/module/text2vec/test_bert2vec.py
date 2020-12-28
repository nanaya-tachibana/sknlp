from collections import Counter

import numpy as np

from sknlp.vocab import Vocab
from sknlp.module.text2vec import Bert2vec

from .test_text2vec import TestText2vec


class TestBert2vec(TestText2vec):

    vocab = Vocab(Counter({"你": 10, "我": 12, "他": 22, "好": 221, "坏": 12}),
                  unk_token="[UNK]",
                  pad_token="[PAD]")
    segmenter = None
    hidden_size = 768
    sequence_length = None
    max_sequence_length = 100
    num_layers = 3
    module = Bert2vec(
        vocab,
        segmenter=segmenter,
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
            self.module(np.array(["你好"]))[0],
            new_module(np.array(["你好"]))[0]
        )
