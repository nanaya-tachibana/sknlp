from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K


class WeightDecayScheduler(Callback):
    def __init__(self, warmup_schedule, decay_schedule, verbose=0):
        super().__init__()
        self.warmup_schedule = warmup_schedule
        self.decay_schedule = decay_schedule
        self.epoch = 0
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        if not hasattr(self.model.optimizer, "weight_decay"):
            raise ValueError('Optimizer must have a "weight_decay" attribute.')
        wd = float(K.get_value(self.model.optimizer.weight_decay))
        new_wd = self.decay_schedule(epoch, wd)
        if new_wd == wd:
            return
        K.set_value(self.model.optimizer.weight_decay, K.get_value(new_wd))
        if self.verbose > 1:
            print(
                "\nEpoch %05d: WeightDecayScheduler reducing weight decay "
                "to %s." % (epoch + 1, new_wd)
            )

    def on_train_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, "weight_decay"):
            raise ValueError('Optimizer must have a "weight_decay" attribute.')
        wd = float(K.get_value(self.model.optimizer.weight_decay))
        step = self.epoch * self.params["steps"] + batch
        new_wd = self.warmup_schedule(step, wd)
        if new_wd == wd:
            return
        K.set_value(self.model.optimizer.weight_decay, K.get_value(new_wd))
        if self.verbose > 1:
            print(
                "\nStep %05d: WeightDecayScheduler increasing weight decay "
                "to %s." % (step + 1, new_wd)
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["weight_decay"] = K.get_value(self.model.optimizer.weight_decay)
