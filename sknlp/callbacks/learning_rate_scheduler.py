from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K


class LearningRateScheduler(Callback):
    def __init__(self, warmup_schedule, decay_schedule, verbose=0):
        super().__init__()
        self.warmup_schedule = warmup_schedule
        self.decay_schedule = decay_schedule
        self.epoch = 0
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs):
        self.epoch = epoch
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        new_lr = self.decay_schedule(epoch, lr)
        if new_lr == lr:
            return
        K.set_value(self.model.optimizer.lr, K.get_value(new_lr))
        if self.verbose > 0:
            print(
                "\nEpoch %05d: LearningRateScheduler decreasing learning rate "
                "to %s." % (epoch + 1, new_lr)
            )

    def on_train_batch_begin(self, batch, logs):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        step = self.epoch * self.params["steps"] + batch
        new_lr = self.warmup_schedule(step, lr)
        if new_lr == lr:
            return
        K.set_value(self.model.optimizer.lr, K.get_value(new_lr))
        if self.verbose > 0:
            print(
                "\nStep %07d: LearningRateScheduler increasing learning rate "
                "to %s." % (step + 1, new_lr)
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = K.get_value(self.model.optimizer.lr)