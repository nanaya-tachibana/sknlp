from typing import List, Tuple, Dict, Union, Optional
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit, softmax


# (精准率, 召回率, F值, 数量)
FscoreTuple = Tuple[float, float, float, Optional[int]]


def logits2scores(logits: np.ndarray, is_multilabel: bool) -> np.ndarray:
    """
    logits转换为sigmoid或softmax分值.

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
            'logits should have shape(`n_samples`, `n_classes`),'
            'but shape%s was given' % str(logits.shape)
        )
    if is_multilabel:
        return expit(logits)
    else:
        return softmax(logits, axis=1)


def scores2classes(
    scores: np.ndarray, is_multilabel: bool,
    threshold: Union[np.ndarray, float] = 0.5
) -> Union[List[int], List[List[int]]]:
    """
    根据给定的`threshold`, 将分类问题的`scores`解析为对应的类别.

    Parameters
    ----------
    scores: 输入的`logits`, shape(`n_samples`, `n_classes`)
    is_multilabel: 是否为多标签分类
    threshold: 正例判断的阈值(仅在多标签和二分类问题时有作用)

    Returns
    ----------
    返回一个长度为`n_samples`的``list``.

    如果是多标签分类, 每一个sample对应的结果为一个``list``,
    其中的每个``int``值为这个sample对应的类别.

    如果不是多标签分类, 则每一个sample对应的结果为一个``int``,
    表示这个sample对应的类别.
    """
    assert scores.ndim == 2
    if is_multilabel:
        return [
            np.where(scores[i, :] > threshold)[0].tolist()
            for i in range(scores.shape[0])
        ]
    else:
        return np.argmax(scores, axis=1).tolist()


def logits2classes(
    logits: np.ndarray, is_multilabel: bool,
    threshold: Union[np.ndarray, float] = 0.5
) -> Union[List[int], List[List[int]]]:
    """
    根据给定的`threshold`, 将分类问题的`logits`解析为对应的类别.

    Parameters
    ----------
    logits: 输入的`logits`, shape(`n_samples`, `n_classes`)
    is_multilabel: 是否为多标签分类
    threshold: 正例判断的阈值(仅在多标签和二分类问题时有作用)

    Returns
    ----------
    返回一个长度为`n_samples`的``list``.

    如果是多标签分类, 每一个sample对应的结果为一个``list``,
    其中的每个``int``值为这个sample对应的类别.

    如果不是多标签分类, 则每一个sample对应的结果为一个``int``,
    表示这个sample对应的类别.
    """
    assert logits.ndim == 2
    num_classes = logits.shape[1]
    if is_multilabel:
        return [
            np.where(expit(logits[i, :]) > threshold)[0].tolist()
            for i in range(logits.shape[0])
        ]
    elif num_classes == 2:
        return np.where(expit(logits[:, 1]) > threshold, 1, 0).tolist()
    else:
        return np.argmax(logits, axis=1).tolist()


def classify_f_score(
    y: List[List[int]], p: List[List[int]], is_multilabel: bool,
    beta: float = 1, labels: Optional[List[str]] = None
) -> Dict[Union[str, int], FscoreTuple]:
    """
    计算分类结果的F值.

    Parameters
    ----------
    y: 标注标签
    p: 预测标签
    is_multilabel: 是否是多标签分类
    beta: 加权值, 默认为1, f = (1 + beta**2) * (P * R) / (beta**2 * P + R)
    labels: 标签名, 如果不提供则取0...n_classes - 1

    Returns
    ----------
    返回一个dict.

    如果是非多标签的二分类问题, 则仅包含score一个key, value是一个Tuple
    是正例的(precision, recall, f score, num samples).

    其他情况下, key是一个类别或者avg, value是一个Tuple,
    是对应key的(precision, recall, f score, num samples).
    """
    scores: Dict[Union[str, int], FscoreTuple] = dict()
    _y, _p = np.array(y), np.array(p)
    assert _y.shape == _p.shape, 'y and p must have same shape'

    max_class_idx = _y.max()
    if max_class_idx <= 1 and not is_multilabel:
        score = precision_recall_fscore_support(_y, _p, average='binary')
        scores['score'] = score
    else:
        num_classes = _y.shape[1] if _y.ndim == 2 else max_class_idx + 1
        idx = list(range(num_classes))
        detail_score = precision_recall_fscore_support(_y, _p,
                                                       beta=beta, labels=idx)
        micro_score = precision_recall_fscore_support(
            _y, _p, beta=beta, labels=idx, average='micro'
        )
        for i, score in enumerate(zip(*detail_score)):
            if labels is None:
                scores[i] = score
            else:
                scores[labels[i]] = score
        scores['avg'] = micro_score
    return scores
