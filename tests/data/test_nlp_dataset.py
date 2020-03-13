# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd

from sknlp.data.nlp_dataset import NLPDataset #dataframe_to_dataset,
                                    #_get_csv_dataset,
                                    #_supervised_dataset_transform,
                                    #_make_dataset_from_csv)


def test_dataframe_to_dataset():
    df = pd.DataFrame(zip([1, 2, 3], ['a', 'b', 'c']))
    dataset = NLPDataset.dataframe_to_dataset(df)
    for text, label in dataset:
        assert text.numpy() == 1
        assert label.numpy().decode('utf-8') == 'a'
        break


def test_load_csv(tmp_path):
    test_file = tmp_path / 'test.txt'
    test_file.write_text('text\tlabel\n你啊拿好\t\n我好\t2|1\n')
    # tf.data.experimental.CsvDataset don't support empty value for now
    for in_memory in (True, ):
        dataset, size = NLPDataset.load_csv(str(test_file), '\t', in_memory)
        for text, label in dataset:
            assert text.numpy().decode('utf-8') == '你啊拿好'
            assert label.numpy().decode('utf-8') == ''
            assert size == 2 if in_memory else -1
            break


def test_supervised_dataset_transform():
    nlp_dataset = NLPDataset()
    dataset = tf.data.Dataset.from_tensor_slices((['xxx', 'yyyyy'],
                                                  ['1|2', '2']))
    dataset = nlp_dataset.batchify(nlp_dataset.transform(dataset),
                                   batch_size=2,
                                   shuffle=False)
    for text, label in dataset:
        text = text.numpy()
        assert text[0][-1].decode('utf-8') == '<pad>'
        assert text[1][0].decode('utf-8') == 'y'
        label = label.numpy()
        assert label[0].decode('utf-8') == '1|2'
