import random
import itertools

import pytest

from ..conftest import random_letters, random_digits


@pytest.fixture
def raw_data():
    n = 150
    letters = [random_letters(random.choice([4, 5, 6])) for _ in range(n)]
    digits = [random_digits(random.choice([4, 5, 6])) for _ in range(n)]
    X = itertools.chain(letters[: n // 3], digits[: n // 3])
    y = itertools.chain(
        zip(letters[n // 3 : -n // 3], digits[-n // 3 :]),
        zip(digits[n // 3 : -n // 3], letters[-n // 3 :]),
    )
    X_y = list(zip(X, y))
    random.shuffle(X_y)
    X, y = zip(*X_y)
    training_size = int(n // 3 * 2 * 0.8)
    training_X = X[:training_size]
    training_y = y[:training_size]
    validation_X = X[training_size:]
    validation_y = y[training_size:]

    letter_letter = [
        (
            random_letters(random.choice([4, 5, 6])),
            random_letters(random.choice([4, 5, 6])),
        )
        for _ in range(10)
    ]
    digit_digit = [
        (
            random_digits(random.choice([4, 5, 6])),
            random_digits(random.choice([4, 5, 6])),
        )
        for _ in range(10)
    ]
    digit_letter = [
        (
            random_digits(random.choice([4, 5, 6])),
            random_letters(random.choice([4, 5, 6])),
        )
        for _ in range(20)
    ]
    evaludation_X = list(itertools.chain(letter_letter, digit_digit, digit_letter))
    evaludation_y = list(
        itertools.chain([1 for _ in range(20)], [0 for _ in range(20)])
    )
    return [
        (training_X, training_y),
        (validation_X, validation_y),
        (evaludation_X, evaludation_y),
    ]


@pytest.fixture
def file_data(raw_data, tmp_path_factory):
    (
        (training_X, training_y),
        (validation_X, validation_y),
        (evaludation_X, evaludation_y),
    ) = raw_data
    tmp_path = tmp_path_factory.mktemp("retrieval")
    training_filename = str(tmp_path / "training")
    validation_filename = str(tmp_path / "validation")
    evaluation_filename = str(tmp_path / "evaluation")

    def write_data(filename, X, y):
        with open(filename, "w") as f:
            f.write("text\tpositive\tnegative\n")
            for xi, (y0i, y1i) in zip(X, y):
                f.write(f"{xi}\t{y0i}\t{y1i}\n")

    write_data(training_filename, training_X, training_y)
    write_data(validation_filename, validation_X, validation_y)
    with open(evaluation_filename, "w") as f:
        f.write("text1\ttext2\tlabel\n")
        for (x0i, x1i), yi in zip(evaludation_X, evaludation_y):
            f.write(f"{x0i}\t{x1i}\t{yi}\n")
    return training_filename, validation_filename, evaluation_filename