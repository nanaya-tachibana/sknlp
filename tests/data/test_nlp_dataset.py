import pandas as pd

from sknlp.data.nlp_dataset import NLPDataset


def test_dataframe_to_dataset():
    df = pd.DataFrame({"text": [1, 2, 3], "label": ["a", "b", "c"]})
    dataset = NLPDataset.dataframe_to_dataset(df, ["int", "str"], "")
    for text, label in dataset:
        assert text.numpy() == 1
        assert label.numpy().decode("utf-8") == "a"
        break


def test_load_csv(text_with_empty):
    test_file = text_with_empty
    # tf.data.experimental.CsvDataset don't support empty value for now
    for in_memory in (True,):
        dataset, size = NLPDataset.load_csv(
            str(test_file), "\t", in_memory, ["str", "str", "int"], ""
        )
        for text, label in dataset:
            assert text.numpy().decode("utf-8") == "你啊拿好"
            assert label.numpy().decode("utf-8") == ""
            assert size == 2 if in_memory else -1
            break