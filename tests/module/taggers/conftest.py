import json
import random
import string

import pandas as pd
import pytest


@pytest.fixture
def raw_data():
    def insert_sequence(text, position, sequence, label):
        new_text = "".join([text[:position], sequence, text[position:]])
        tag = [position, position + len(sequence) - 1, label]
        return new_text, tag

    def random_digits(length):
        return "".join(random.choices(string.digits, k=length))

    def random_letters(length):
        return "".join(random.choices(string.ascii_uppercase, k=length))

    def random_sequence():
        length = random.choice([2, 3, 4])
        if random.random() > 0.5:
            return random_digits(length), "digit"
        else:
            return random_letters(length), "letter"

    n = 100
    texts = []
    tags_list = []
    for _ in range(n):
        length = random.choice(range(10, 20))
        text = "".join(random.choices(string.punctuation, k=length))
        num_tags = 1
        if random.random() > 0.5:
            num_tags += 1
        positions = random.choices(range(length), k=num_tags)
        positions.sort()
        offset = 0
        tags = []
        for pos in positions:
            sequence, label = random_sequence()
            text, tag = insert_sequence(text, pos + offset, sequence, label)
            tags.append(tag)
            offset += len(sequence)
        texts.append(text)
        tags_list.append(tags)

    X_y = list(zip(texts, tags_list))
    random.shuffle(X_y)
    texts, tags_list = zip(*X_y)
    training_size = int(n * 0.8)
    training_X = texts[:training_size]
    training_y = tags_list[:training_size]
    validation_X = texts[training_size:]
    validation_y = tags_list[training_size:]
    return [["letter", "digit"], (training_X, training_y), (validation_X, validation_y)]


@pytest.fixture
def file_data(raw_data, tmp_path_factory):
    _, (training_X, training_y), (validation_X, validation_y) = raw_data
    tmp_path = tmp_path_factory.mktemp("tagging")
    training_filename = str(tmp_path / "training")
    validation_filename = str(tmp_path / "validation")

    def write_data(filename, X, y):
        y = [json.dumps(yi) for yi in y]
        df = pd.DataFrame(zip(X, y), columns=["text", "label"])
        df.to_csv(filename, sep="\t", index=None, quoting=3, escapechar="\\")

    write_data(training_filename, training_X, training_y)
    write_data(validation_filename, validation_X, validation_y)
    return training_filename, validation_filename