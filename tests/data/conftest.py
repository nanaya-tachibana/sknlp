import pytest

from sknlp.vocab import Vocab


@pytest.fixture
def text_without_empty(tmp_path):
    test_file = tmp_path / "noempty.txt"
    test_file.write_text("text\tlabel\n你啊拿好\t1\n我好\t2|1\n")
    return str(test_file)


@pytest.fixture
def text_with_empty(tmp_path):
    test_file = tmp_path / "withempty.txt"
    test_file.write_text("text\tlabel\n你啊拿好\t\n我好\t2|1\n")
    return str(test_file)


@pytest.fixture
def vocab():
    return Vocab("甲乙丙丁葵", bos_token="[CLS]", eos_token="[SEP]")