import random
import string

import pytest

from sknlp.module.text2vec import Bert2vec, Word2vec


def random_digits(length):
    return "".join(random.choices(string.digits, k=length))


def random_letters(length):
    return "".join(random.choices(string.ascii_uppercase, k=length))


@pytest.fixture
def model_common_test(tmp_path):
    def test_model(
        model_class, model, raw_data, file_data, use_raw_data, learning_rate
    ):
        (training_X, training_y), (validation_X, validation_y) = raw_data
        training_filename, validation_filename = file_data

        save_path = str(tmp_path / "saved")
        log_file = str(tmp_path / "log.txt")
        kwargs = dict(
            batch_size=32,
            enable_early_stopping=True,
            early_stopping_monitor=2,
            early_stopping_use_best_epoch=True,
            learning_rate_update_epochs=3,
            learning_rate_warmup_steps=3,
            learning_rate=learning_rate,
            n_epochs=20,
            weight_decay=1e-8,
            checkpoint=save_path,
            log_file=log_file,
            verbose=0,
        )
        if not use_raw_data:
            kwargs["dataset"] = model.create_dataset_from_csv(training_filename)
            kwargs["validation_dataset"] = model.create_dataset_from_csv(
                validation_filename
            )
        else:
            kwargs.update(
                dict(
                    X=training_X,
                    y=training_y,
                    validation_X=validation_X,
                    validation_y=validation_y,
                )
            )
        model.fit(**kwargs)

        model.save(save_path)
        model = model_class.load(save_path, epoch=0)
        model = model_class.load(save_path)
        if use_raw_data:
            model.predict(X=validation_X)
        else:
            model.predict(dataset=kwargs["dataset"])
        if use_raw_data:
            model.score(X=validation_X, y=validation_y)
        else:
            model.score(dataset=kwargs["validation_dataset"])
        model.export(str(tmp_path), "export", "0")

    return test_model


@pytest.fixture
def bert2vec():
    return Bert2vec.from_tfv1_checkpoint(1, "data/bert_3l")


@pytest.fixture
def word2vec():
    return Word2vec.from_word2vec_format("data/char/vec.txt", segmenter="char")