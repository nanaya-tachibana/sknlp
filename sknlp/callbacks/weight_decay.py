from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K


class WeightDecayScheduler(Callback):
    def __init__(self, schedule, verbose=0):
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "weight_decay"):
            raise ValueError('Optimizer must have a "weight_decay" attribute.')
        wd = float(K.get_value(self.model.optimizer.weight_decay))
        wd = self.schedule(epoch, wd)
        K.set_value(self.model.optimizer.weight_decay, K.get_value(wd))
        if self.verbose > 0:
            print(
                "\nEpoch %05d: WeightDecayScheduler reducing weight decay "
                "to %s." % (epoch + 1, wd)
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["weight_decay"] = K.get_value(self.model.optimizer.weight_decay)
