import functools
import jieba_fast as jieba


def get_segmenter(name):
    if name == 'jieba':
        return functools.partial(jieba.lcut, HMM=False)
    elif name == 'list' or name == 'char':
        return list
    elif name is None:
        return lambda x: x
    else:
        raise ValueError('unknown segmenter %s' % name)
