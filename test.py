from __future__ import annotations
import itertools
import token
import jieba_fast as jieba
import pandas as pd

import tensorflow as tf

from sknlp.module.generators import BertGenerator


if __name__ == "__main__":
    gen = BertGenerator.load("data/temp")
    dataset = gen.create_dataset_from_csv("data/gen_temp.csv")
    print(gen.format_score(gen.score(dataset=dataset, batch_size=50)))