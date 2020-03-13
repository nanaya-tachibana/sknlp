from sknlp.module.text_rnn import TextRNN
from tests.module.test_base_model import TestSupervisedNLPModel


class TestTextRNN(TestSupervisedNLPModel):

    model = TextRNN(['1', '2', '3'], rnn_recurrent_dropout=0.2)

    def test_config(self):
        super().test_config()
        assert self.model.get_config()['rnn_recurrent_dropout'] == 0.2
