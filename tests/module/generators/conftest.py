import random

import pandas as pd
import pytest

from ..conftest import random_letters


@pytest.fixture
def raw_data():
    n = 100
    texts = []
    reversed_texts = []
    for _ in range(n):
        length = random.choice(range(3, 6))
        text = random_letters(length)
        reversed_text = text[::-1]
        texts.append(text)
        reversed_texts.append(reversed_text)

    training_size = int(n * 0.9)
    training_X = texts[:training_size]
    training_y = reversed_texts[:training_size]
    validation_X = texts[training_size:]
    validation_y = reversed_texts[training_size:]
    return [(training_X, training_y), (validation_X, validation_y)]


@pytest.fixture
def file_data(raw_data, tmp_path_factory):
    (training_X, training_y), (validation_X, validation_y) = raw_data
    tmp_path = tmp_path_factory.mktemp("generation")
    training_filename = str(tmp_path / "training")
    validation_filename = str(tmp_path / "validation")

    def write_data(filename, X, y):
        df = pd.DataFrame(zip(X, y), columns=["text", "label"])
        df.to_csv(filename, sep="\t", index=None, quoting=3, escapechar="\\")

    write_data(training_filename, training_X, training_y)
    write_data(validation_filename, validation_X, validation_y)
    return training_filename, validation_filename