import os
import tempfile

import pandas as pd

from sknlp.module.text2vec import Bert2vec, BertFamily
from sknlp.module.retrievers import BertRetriever


BERT_CHECKPOINT_PATH = "RoBERTa-tiny3L768-clue"
BERT_TYPE = BertFamily.BERT
ENABLE_RECOMPUTE_GRAD = False
BATCH_SIZE = 120
HAS_NEGATIVE = True
UNSUPERVISED = False
# 不过除了PAWSX之外，其他4个任务都不需要全部数据都拿来训练，
# 经过测试，只需要随机选1万个训练样本训练一个epoch即可训练到最有效果（更多样本更少样本效果都变差）。
# 自苏剑林. (Apr. 26, 2021). 《中文任务还是SOTA吗？我们给SimCSE补充了一些实验 》[Blog post].
# Retrieved from https://kexue.fm/archives/8348
NUM_UNSUPERVISED_SAMPLES = 10000

DIR = "LCQMC"
TRAINING_FILENAME = os.path.join(DIR, "train.txt")
VALIDATION_FILENAME = os.path.join(DIR, "dev.txt")
EVALUATION_FILENAME = os.path.join(DIR, "test.txt")

max_length = 100
with tempfile.TemporaryDirectory() as temp_dir:
    columns = ["text1", "text2", "label"]
    print("读取数据...")
    tvdf = pd.concat(
        [
            pd.read_csv(TRAINING_FILENAME, sep="\t", header=None, quoting=3),
            pd.read_csv(VALIDATION_FILENAME, sep="\t", header=None, quoting=3),
        ]
    )
    tvdf.columns = columns
    text_p99_length = pd.concat(
        [tvdf.text1.apply(lambda x: len(x)), tvdf.text2.apply(lambda x: len(x))]
    ).quantile(0.99)
    max_length = int(min(max_length, text_p99_length))
    edf = pd.read_csv(EVALUATION_FILENAME, sep="\t", header=None, quoting=3)
    edf.columns = columns

    temp_training_validation_filename = os.path.join(temp_dir, "tv.txt")
    tvdf.to_csv(
        temp_training_validation_filename,
        index=None,
        sep="\t",
        quoting=3,
        escapechar="\\",
    )
    temp_evaluation_filename = os.path.join(temp_dir, "e.txt")
    edf.to_csv(
        temp_evaluation_filename,
        index=None,
        sep="\t",
        quoting=3,
        escapechar="\\",
    )

    b2v = Bert2vec.from_tfv1_checkpoint(
        BERT_TYPE, BERT_CHECKPOINT_PATH, enable_recompute_grad=ENABLE_RECOMPUTE_GRAD
    )
    model = BertRetriever(
        has_negative=HAS_NEGATIVE and not UNSUPERVISED,
        max_sequence_length=max_length,
        dropout=0.3,
        text2vec=b2v,
    )

    if not UNSUPERVISED:
        print("生成triple数据...")
        training_filename = os.path.join(temp_dir, "training.csv")
        validation_filename = os.path.join(temp_dir, "validation.csv")
        dataset = model.create_dataset_from_01_label_dataset(
            temp_training_validation_filename
        )
        validation_size = int(edf.shape[0])
        pd.DataFrame(
            list(zip(dataset.X[:-validation_size], *zip(*dataset.y[:-validation_size])))
        ).to_csv(
            training_filename,
            index=None,
            sep="\t",
            escapechar="\\",
            quoting=3,
        )
        pd.DataFrame(
            list(zip(dataset.X[-validation_size:], *zip(*dataset.y[-validation_size:])))
        ).sample(validation_size).to_csv(
            validation_filename,
            index=None,
            sep="\t",
            escapechar="\\",
            quoting=3,
        )

    kwargs = dict()

    training_size = None
    validation_size = None
    if UNSUPERVISED:
        X = (
            pd.concat([tvdf.text1, tvdf.text2])
            .sample(tvdf.shape[0] * 2)
            .values.tolist()[:NUM_UNSUPERVISED_SAMPLES]
        )
        validation_X = (
            pd.concat([edf.text1, edf.text2]).sample(edf.shape[0] * 2).values.tolist()
        )
        training_size = len(X)
        validation_size = len(validation_X)
        kwargs.update(X=X, validation_X=validation_X)
    elif HAS_NEGATIVE:
        dataset = model.create_dataset_from_csv(training_filename)
        validation_dataset = model.create_dataset_from_csv(validation_filename)
        training_size = len(dataset.X)
        validation_size = len(validation_dataset.X)
        kwargs.update(dataset=dataset, validation_dataset=validation_dataset)
    else:
        tdf = pd.read_csv(training_filename, sep="\t", escapechar="\\", quoting=3)
        tdf = tdf[tdf[0] != tdf[1]]
        vdf = pd.read_csv(validation_filename, sep="\t", escapechar="\\", quoting=3)
        vdf = vdf[vdf[0] != vdf[1]]
        training_size = tdf.shape[0]
        validation_size = vdf.shape[0]
        kwargs.update(
            X=tdf[0].values.tolist(),
            y=tdf[1].values.tolist(),
            validation_X=vdf[0].values.tolist(),
            validation_y=vdf[1].values.tolist(),
        )
    print(f"训练集大小: {training_size}, 验证集大小: {validation_size}, 文本截断长度: {max_length}")
    print("开始训练...")
    model.fit(
        batch_size=BATCH_SIZE,
        n_epochs=15,
        learning_rate=2e-4,
        learning_rate_update_epochs=2,
        learning_rate_warmup_steps=training_size // BATCH_SIZE * 8 // 10,
        weight_decay=1e-2,
        enable_early_stopping=True,
        early_stopping_monitor=1,
        early_stopping_patience=3,
        checkpoint="./bq_bert",
        log_file="./bq_bert.log",
        **kwargs,
    )
    model.save("./bq_bert")
    model = BertRetriever.load("./bq_bert")
    print(
        model.format_score(
            model.score(
                X=list(zip(edf.text1, edf.text2)),
                y=edf.label.values.tolist(),
                batch_size=BATCH_SIZE,
            )
        )
    )