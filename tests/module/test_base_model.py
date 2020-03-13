from sknlp.module.base_model import BaseNLPModel, SupervisedNLPModel


def test_build_vocab():
    texts = ['11111', '222', '33333']
    vocab = BaseNLPModel.build_vocab(texts, list, min_frequency=4)
    assert len(vocab) == 6
    assert '1' in vocab
    assert '2' not in vocab


class TestSupervisedNLPModel:

    model = SupervisedNLPModel(['1', '2', '3'])

    def test_config(self):
        assert self.model.get_config()['classes'] == ['1', '2', '3']
