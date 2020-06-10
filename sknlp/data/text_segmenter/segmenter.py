import functools
import jieba_fast as jieba


def get_segmenter(name):
    if name == 'jieba':
        return functools.partial(jieba.lcut, HMM=False)
    elif name == 'list' or name == 'char':
        return list
    else:
        raise ValueError('unknown segmenter %s' % name)
