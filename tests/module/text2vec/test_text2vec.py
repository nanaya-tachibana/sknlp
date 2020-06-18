from collections import Counter

from sknlp.vocab import Vocab
from sknlp.module.text2vec import Text2vec


class TestText2vec:

    vocab = Vocab(Counter({'a': 10, 'b': 12, 'c': 22}))
    segmenter = 'jieba'
    module = Text2vec(vocab, segmenter)

    def test_vocab(self):
        assert self.module.vocab['a'] == self.vocab['a']

    def test_segmeter(self):
        assert self.module.segmenter == self.segmenter

    def test_save_vocab(self, tmp_path):
        filename = "vocab.json"
        tmp_file = tmp_path / filename
        self.module.save_vocab(tmp_path, filename=filename)
        with open(tmp_file) as f:
            vocab = Vocab.from_json(f.read())
        assert vocab["a"] == self.module.vocab["a"]

    def test_get_config(self):
        config = self.module.get_config()
        assert config["segmenter"] == self.segmenter
