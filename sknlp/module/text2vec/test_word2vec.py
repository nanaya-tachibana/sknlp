import pathlib
import numpy as np

from sknlp.vocab import Vocab
from .word2vec import Word2vec


def test_word2vec():
    vocab = Vocab(["a", "b", "c"])
    wv = Word2vec(vocab, 10)
    token_ids = np.array([1, 2, 0, 0])
    embeddings = wv(token_ids)
    assert embeddings.shape == (4, 10)
    assert embeddings._keras_mask.numpy().tolist() == [True, True, False, False]


def test_load_word2vec_format(tmp_path):
    filepath: pathlib.Path = tmp_path / "vec.txt"
    with open(str(filepath), "w") as f:
        f.write("4 3\n")
        f.write(" ".join(["a", " ".join(["0.1"] * 3)]))
        f.write("\n")
        f.write(" ".join(["<unk>", " ".join(["0.3"] * 3)]))
        f.write("\n")
        f.write(" ".join(["b", " ".join(["0.2"] * 3)]))
        f.write("\n")
        f.write(" ".join(["</s>", " ".join(["0.4"] * 3)]))
    wv = Word2vec.from_word2vec_format(str(filepath))
    assert len(wv.vocab) == 6
    assert wv.vocab["a"] == 4
    assert wv.vocab["</s>"] == 3
    np.testing.assert_array_almost_equal(wv._model.get_weights()[0][1, :], [0.3] * 3)
    np.testing.assert_array_almost_equal(wv._model.get_weights()[0][4, :], [0.1] * 3)


def test_save_load_glove_format(tmp_path):
    filepath: pathlib.Path = tmp_path / "vec.txt"
    with open(str(filepath), "w") as f:
        f.write(" ".join(["a", " ".join(["0.1"] * 3)]))
        f.write("\n")
        f.write(" ".join(["<s>", " ".join(["0.3"] * 3)]))
        f.write("\n")
        f.write(" ".join(["b", " ".join(["0.2"] * 3)]))
        f.write("\n")
        f.write(" ".join(["</s>", " ".join(["0.4"] * 3)]))
    wv = Word2vec.from_word2vec_format(str(filepath))
    assert len(wv.vocab) == 6
    assert wv.vocab["a"] == 4
    assert wv.vocab["</s>"] == 3
    np.testing.assert_array_almost_equal(wv._model.get_weights()[0][1, :], [0.15] * 3)
    np.testing.assert_array_almost_equal(wv._model.get_weights()[0][4, :], [0.1] * 3)

    wv.to_word2vec_format(filepath)
    new_wv = Word2vec.from_word2vec_format(filepath)
    assert len(wv.vocab) == len(new_wv.vocab)
    np.testing.assert_array_almost_equal(wv._model.weights[0], new_wv._model.weights[0])