from __future__ import annotations
from typing import Optional, Any
import tensorflow as tf

from sknlp.utils.tagging import (
    convert_ids_to_tags,
    tagging_fscore,
)
from sknlp.utils.logging import logger


class TaggingFScoreMetric(tf.keras.callbacks.Callback):
    def __init__(self, classes: list[str], add_start_end_tag: bool) -> None:
        super().__init__()
        self.classes = classes
        self.add_start_end_tag = add_start_end_tag

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        if self.validation_data is None:
            return

        idx2class = dict(zip(range(len(self.classes)), self.classes))
        tag_ids_list = self.model.predict(self.validation_data)
        predictions = []
        for tag_ids in tag_ids_list:
            predictions.append(
                convert_ids_to_tags(
                    tag_ids.numpy().tolist(),
                    lambda x: idx2class[x],
                    add_start_end_tag=self.add_start_end_tag,
                )
            )

        y = []
        for (data,) in self.validation_data.as_numpy_iterator():
            tag_ids_array = data[-1]
            y.extend(
                [
                    convert_ids_to_tags(
                        tag_ids,
                        lambda x: idx2class[x],
                        add_start_end_tag=self.add_start_end_tag,
                    )
                    for tag_ids in tag_ids_array.tolist()
                ]
            )

        score_df = tagging_fscore(y, predictions, self.classes[1:])
        logger.info(score_df)