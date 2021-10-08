from .bert_converter import (
    BertCheckpointConverter,
    AlbertCheckpointConverter,
    ElectraCheckpointConverter,
)


def test_bert_converter():
    converter = BertCheckpointConverter(3, 12, "bert2vec")
    converter.convert("data/bert_3l")


def test_albert_converter():
    converter = AlbertCheckpointConverter(4, 12, "bert2vec")
    converter.convert("data/albert_tiny_zh_google")


def test_electra_converter():
    converter = ElectraCheckpointConverter(12, 4, "")
    converter.convert("data/electra_180g_small")
