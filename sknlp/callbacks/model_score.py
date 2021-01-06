from typing import Dict, Any, Optional, Callable
import logging
import tensorflow as tf


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
logger.addHandler(stream)


class ModelScoreCallback(tf.keras.callbacks.Callback):

    def __init__(self, score_func: Callable):
        super().__init__()
        self._score_func = score_func

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        score_df = self._score_func()
        row = score_df[score_df["class"] == "avg"]
        logger.info(row)
        for col in score_df.columns:
            if col == "support":
                continue
            logs[f"val_{col}"] = row[col].values.tolist()[0]
