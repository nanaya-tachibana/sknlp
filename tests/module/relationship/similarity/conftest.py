import random

import pytest

from ..conftest import random_letters, random_digits


@pytest.fixture
def raw_data():
    n = 100
    letters = [random_letters(random.choice([4, 5, 6])) for _ in range(n)]
    digits = [random_digits(random.choice([4, 5, 6])) for _ in range(n)]
    letter_letter = zip(letters[: n // 4], letters[n // 4 : n // 2])
    digit_digit = zip(digits[: n // 4], digits[n // 4 : n // 2])
    letter_digit = zip(letters[n // 2 : n - n // 4], digits[n // 2 : n - n // 4])
    digit_letter = zip(digits[-n // 4 :], letters[-n // 4 :])
    X = [*letter_letter, *digit_digit, *letter_digit, *digit_letter]
    y = [*[1 for _ in range(n // 2)], *[0 for _ in range(n // 2)]]
    X_y = list(zip(X, y))
    random.shuffle(X_y)
    X, y = zip(*X_y)
    training_size = int(n * 0.8)
    training_X = X[:training_size]
    training_y = y[:training_size]
    validation_X = X[training_size:]
    validation_y = y[training_size:]
    return [
        (training_X, training_y),
        (validation_X, validation_y),
    ]


@pytest.fixture
def file_data(raw_data, tmp_path_factory):
    (training_X, training_y), (validation_X, validation_y) = raw_data
    tmp_path = tmp_path_factory.mktemp("similarity")
    training_filename = str(tmp_path / "training")
    validation_filename = str(tmp_path / "validation")

    def write_data(filename, X, y):
        with open(filename, "w") as f:
            f.write("text\tcontext\tlabel\n")
            for (x0i, x1i), yi in zip(X, y):
                f.write(f"{x0i}\t{x1i}\t{yi}\n")

    write_data(training_filename, training_X, training_y)
    write_data(validation_filename, validation_X, validation_y)
    return training_filename, validation_filename