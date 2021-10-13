import random
import string

import pytest


@pytest.fixture
def raw_data():
    num_letters, num_digits = 100, 20
    size = num_letters + num_digits
    length = random.choices([4, 5, 6], k=size)
    letters = [
        "".join(random.choices(string.ascii_uppercase, k=k))
        for k in length[:num_letters]
    ]
    digits = ["".join(random.choices(string.digits, k=k)) for k in length[num_letters:]]
    training_X = [*letters[: int(num_letters * 0.8)], *digits[: int(num_digits * 0.8)]]
    training_size = int(size * 0.8)
    training_y = [
        "letter" if i < num_letters * 0.8 else "digit" for i in range(training_size)
    ]
    validation_X = [
        *letters[int(num_letters * 0.8) :],
        *digits[int(num_digits * 0.8) :],
    ]
    validation_size = int(size * 0.2)
    validation_y = [
        "letter" if i < num_letters * 0.2 else "digit" for i in range(validation_size)
    ]
    return [["letter", "digit"], (training_X, training_y), (validation_X, validation_y)]


@pytest.fixture
def file_data(raw_data, tmp_path_factory):
    _, (training_X, training_y), (validation_X, validation_y) = raw_data
    tmp_path = tmp_path_factory.mktemp("classification")
    training_filename = str(tmp_path / "training")
    validation_filename = str(tmp_path / "validation")

    def write_data(filename, X, y):
        with open(filename, "w") as f:
            f.write("text\tlabel\n")
            for xi, yi in zip(X, y):
                f.write(f"{xi}\t{yi}\n")

    write_data(training_filename, training_X, training_y)
    write_data(validation_filename, validation_X, validation_y)
    return training_filename, validation_filename


@pytest.fixture
def raw_data_pairwise():
    n = 100
    letters = [
        "".join(random.choices(string.ascii_uppercase, k=random.choice([4, 5, 6])))
        for _ in range(n)
    ]
    digits = [
        "".join(random.choices(string.digits, k=random.choice([4, 5, 6])))
        for _ in range(n)
    ]
    letter_letter = zip(letters[: n // 4], letters[n // 4 : n // 2])
    digit_digit = zip(digits[: n // 4], digits[n // 4 : n // 2])
    letter_digit = zip(letters[n // 2 : n - n // 4], digits[n // 2 : n - n // 4])
    digit_letter = zip(digits[-n // 4 :], letters[-n // 4 :])
    X = [*letter_letter, *digit_digit, *letter_digit, *digit_letter]
    y = [*["same" for _ in range(n // 2)], *["different" for _ in range(n // 2)]]
    X_y = list(zip(X, y))
    random.shuffle(X_y)
    X, y = zip(*X_y)
    training_size = int(n * 0.8)
    training_X = X[:training_size]
    training_y = y[:training_size]
    validation_X = X[training_size:]
    validation_y = y[training_size:]
    return [
        ["same", "different"],
        (training_X, training_y),
        (validation_X, validation_y),
    ]


@pytest.fixture
def file_data_pairwise(raw_data_pairwise, tmp_path_factory):
    _, (training_X, training_y), (validation_X, validation_y) = raw_data_pairwise
    tmp_path = tmp_path_factory.mktemp("classification")
    training_filename = str(tmp_path / "training_pairwise")
    validation_filename = str(tmp_path / "validation_pairwise")

    def write_data(filename, X, y):
        with open(filename, "w") as f:
            f.write("text\tcontext\tlabel\n")
            for (x0i, x1i), yi in zip(X, y):
                f.write(f"{x0i}\t{x1i}\t{yi}\n")

    write_data(training_filename, training_X, training_y)
    write_data(validation_filename, validation_X, validation_y)
    return training_filename, validation_filename