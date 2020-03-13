import functools
from tensorflow.keras.metrics import Accuracy
import tensorflow.keras.backend as K


class AccuracyWithLogits(Accuracy):

    def __init__(self,
                 name='acc',
                 dtype=None,
                 logits2scores='sigmoid'):
        super().__init__(name=name,
                         dtype=dtype)
        self.logits2scores = logits2scores
        if logits2scores == 'sigmoid':
            self._l2s = K.sigmoid
        elif logits2scores == 'softmax':
            self._l2s = functools.partial(K.softmax, axis=-1)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, self._l2s(y_pred),
                             sample_weight=sample_weight)

    def get_config(self):
        return {**super().get_config(), 'logits2scores': self.logits2scores}
