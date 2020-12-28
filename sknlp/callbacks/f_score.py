import logging
import tensorflow as tf


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
logger.addHandler(stream)


def f_score(p, r, beta):
    return (1 + beta ** 2) * (p * r) / (beta ** 2 * p + r) if p + r != 0 else 0


class FScore(tf.keras.callbacks.Callback):

    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def on_epoch_end(self, epoch, logs=None):
        p, r = logs['precision'], logs['recall']
        if 'val_precision' in logs and 'val_recall' in logs:
            val_p, val_r = logs['val_precision'], logs['val_recall']
            logs['val_f-score'] = f_score(val_p, val_r, self.beta)
        else:
            logs['val_f-score'] = 0
        logs['f-score'] = f_score(p, r, self.beta)
        logger.info('f-score: %.2f, val_f-score: %.2f' % (
            logs['f-score'] * 100, logs['val_f-score'] * 100
        ))
