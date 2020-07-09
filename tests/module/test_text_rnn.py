from sknlp.module.text_rnn import TextRNN
from .test_supervised_model import TestSupervisedNLPModel


class TestTextRNN(TestSupervisedNLPModel):

    classes = ['1', '2', '3']
    name = "rr"
    word2vec = None
    segmenter = "jieba"
    max_sequence_length = 100
    model = TextRNN(
        classes,
        max_sequence_length=max_sequence_length,
        segmenter=segmenter,
        text2vec=word2vec,
        rnn_recurrent_dropout=0.2,
        name=name
    )

    def test_get_config(self):
        super().test_get_config()
        assert self.model.get_config()['rnn_recurrent_dropout'] == 0.2
