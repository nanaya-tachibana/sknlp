from __future__ import annotations
from typing import Callable, Optional, Any
import logging
import tensorflow as tf

from sknlp.utils.tagging import (
    convert_ids_to_tags,
    tagging_fscore,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
logger.addHandler(stream)


class TaggingFScoreMetric(tf.keras.callbacks.Callback):
    def __init__(self, classes: list[str]) -> None:
        super().__init__()
        self.classes = classes

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        if self.validation_data is None:
            return

        idx2class = dict(zip(range(len(self.classes)), self.classes))
        tag_ids_list = self.model.predict(self.validation_data)
        predictions = []
        for tag_ids in tag_ids_list:
            predictions.append(
                convert_ids_to_tags(tag_ids.numpy().tolist(), lambda x: idx2class[x])
            )

        y = []
        for _, tag_ids_array in self.validation_data.as_numpy_iterator():
            y.extend(
                [
                    convert_ids_to_tags(tag_ids, lambda x: idx2class[x])
                    for tag_ids in tag_ids_array.tolist()
                ]
            )

        score_df = tagging_fscore(y, predictions, self.classes[1:])
        logger.info(score_df)
        row = score_df[score_df["class"] == "avg"]
        for col in score_df.columns:
            if col == "support":
                continue
            logs[f"val_{col}"] = row[col].values.tolist()[0]