from typing import List, Tuple, Union, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import multilabel_confusion_matrix
from scipy.special import expit, softmax


def logits2probabilities(logits: np.ndarray, is_multilabel: bool) -> np.ndarray:
    """
    计算logits的sigmoid或softmax分值.

    Parameters
    ----------
    logits: 输入的`logits`, shape(`n_samples`, `n_classes`)
    is_multilabel: 是否为多标签分类, 多标签分值使用sigmoid, 非多标签使用softmax

    Returns
    ----------
    返回一个shape(`n_samples`, `n_classes`)的``ndarray``.
    """
    if logits.ndim != 2:
        raise ValueError(
            "logits should have shape(`n_samples`, `n_classes`),"
            "but shape%s was given" % str(logits.shape)
        )
    if is_multilabel:
        return expit(logits)
    else:
        return softmax(logits, axis=1)


def _validate_thresholds(
    thresholds: Union[Sequence[float], float], length: int
) -> List[float]:
    if isinstance(thresholds, float):
        return [thresholds for _ in range(length)]
    elif len(thresholds) != length:
        raise ValueError(
            "thresholds should have length %d, "
            "but length %d was given" % (length, len(thresholds))
        )
    return list(thresholds)


def probabilities2classes(
    probabilities: np.ndarray,
    is_multilabel: bool,
    thresholds: Union[Sequence[float], float] = 0.5,
) -> Union[List[int], List[List[int]]]:
    """
    根据给定的`threshold`, 将分类问题的`scores`解析为对应的类别.

    Parameters
    ----------
    scores: 输入的`logits`, shape(`n_samples`, `n_classes`)
    is_multilabel: 是否为多标签分类
    threshold: 正例判断的阈值(仅在多标签时有作用)

    Returns
    ----------
    返回一个长度为`n_samples`的``list``.

    如果是多标签分类, 每一个sample对应的结果为一个``list``,
    其中的每个``int``值为这个sample对应的类别.

    如果不是多标签分类, 则每一个sample对应的结果为一个``int``,
    表示这个sample对应的类别.
    """
    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities should have shape(`n_samples`, `n_classes`), "
            "but shape%s was given" % str(probabilities.shape)
        )
    thresholds = _validate_thresholds(thresholds, probabilities.shape[1])
    binary: bool = probabilities.shape[1] == 1
    if binary:
        classes: np.ndarray = probabilities >= thresholds
        return classes.reshape(classes.shape[0]).astype("int").tolist()
    if is_multilabel:
        return [
            np.where(probabilities[i, :] > thresholds)[0].tolist()
            for i in range(probabilities.shape[0])
        ]
    else:
        classes = np.argmax(probabilities, axis=1)
        classes[np.all(probabilities <= thresholds, axis=1)] = 0
        return classes.tolist()


def logits2classes(
    logits: np.ndarray,
    is_multilabel: bool,
    thresholds: Union[Sequence[float], float] = 0.5,
) -> Union[List[int], List[List[int]]]:
    """
    根据给定的`threshold`, 将分类问题的`logits`解析为对应的类别.

    Parameters
    ----------
    logits: 输入的`logits`, shape(`n_samples`, `n_classes`)
    is_multilabel: 是否为多标签分类
    threshold: 正例判断的阈值(仅在多标签时有作用)

    Returns
    ----------
    返回一个长度为`n_samples`的``list``.

    如果是多标签分类, 每一个sample对应的结果为一个``list``,
    其中的每个``int``值为这个sample对应的类别.

    如果不是多标签分类, 则每一个sample对应的结果为一个``int``,
    表示这个sample对应的类别.
    """
    if logits.ndim != 2:
        raise ValueError(
            "logits should have shape(`n_samples`, `n_classes`),"
            "but shape%s was given" % str(logits.shape)
        )
    if is_multilabel:
        thresholds = _validate_thresholds(thresholds, logits.shape[1])
        pred = expit(logits)
        return [
            np.where(pred[i, :] > thresholds)[0].tolist()
            for i in range(logits.shape[0])
        ]
    else:
        return np.argmax(logits, axis=1).tolist()


def precision_recall_fscore(
    tp: int, fp: int, fn: int, beta: float = 1
) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    if precision + recall != 0:
        fscore = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    else:
        fscore = 0
    return precision, recall, fscore


def label_binarizer(
    y: Union[Sequence[Sequence[str]], Sequence[str]],
    p: Union[Sequence[Sequence[str]], Sequence[str]],
    classes: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if isinstance(y[0], str):
        y = [[yi] for yi in y]
    if isinstance(p[0], str):
        p = [[pi] for pi in p]
    binarizer = MultiLabelBinarizer(classes=classes)
    return (
        binarizer.fit_transform(y),
        binarizer.transform(p),
        binarizer.classes_.tolist(),
    )


def classification_fscore(
    y: Union[Sequence[Sequence[str]], Sequence[str], Sequence[int]],
    p: Union[Sequence[Sequence[str]], Sequence[str], Sequence[int]],
    classes: Union[Sequence[str], Sequence[int]],
    beta: float = 1,
) -> pd.DataFrame:
    """
    计算分类结果的F值.

    Parameters
    ----------
    y: 标注标签
    p: 预测标签
    classes: 所有的标签集合, 如果不提供则取y
    beta: 加权值, 默认为1, f = (1 + beta**2) * (P * R) / (beta**2 * P + R)

    Returns
    ----------
    返回一个pd.DataFrame.

    """
    if isinstance(y[0], (float, int)):
        confusion_matrix = zip(
            classes, multilabel_confusion_matrix(y, p, labels=classes)
        )
    else:
        y_one_hot, p_one_hot, classes = label_binarizer(y, p, classes)
        confusion_matrix = list(
            zip(classes, multilabel_confusion_matrix(y_one_hot, p_one_hot))
        )
    records = []
    for class_, arr in confusion_matrix:
        tp, fp, fn, tn = arr[1, 1], arr[0, 1], arr[1, 0], arr[0, 0]
        precision, recall, fscore = precision_recall_fscore(tp, fp, fn, beta)
        support = tp + fn
        records.append((class_, precision, recall, fscore, support, tp, fp, fn, tn))

    columns = [
        "class",
        "precision",
        "recall",
        "fscore",
        "support",
        "TP",
        "FP",
        "FN",
        "TN",
    ]
    df = pd.DataFrame(records, columns=columns).sort_values(
        ["support", "TP"], ascending=False
    )

    support, tp, fp, fn, tn = df[["support", "TP", "FP", "FN", "TN"]].sum(axis=0)
    precision, recall, fscore = precision_recall_fscore(tp, fp, fn, beta)
    return df.append(
        pd.DataFrame(
            [("avg", precision, recall, fscore, support, tp, fp, fn, tn)],
            columns=columns,
        )
    )
