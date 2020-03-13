import numpy as np
import tensorflow as tf


class ModelSave(tf.keras.callbacks.Callback):

    def __init__(self, model, dataset):
        super().__init__()
        self.model = model
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):

        np.save('arr-%d' % epoch, self.model.predict(self.dataset))
