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
    def __init__(
        self,
        idx2tag: Callable[[int], str],
        labels: list[str],
        pad_tag: str,
        start_tag: Optional[str] = None,
        end_tag: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.idx2tag = idx2tag
        self.labels = labels
        self.pad_tag = pad_tag
        self.start_tag = start_tag
        self.end_tag = end_tag

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        if self.validation_data is None:
            return

        tag_ids_list = self.model.predict(self.validation_data)
        predictions = []
        for tag_ids in tag_ids_list:
            predictions.append(
                convert_ids_to_tags(
                    self.idx2tag,
                    tag_ids.numpy().tolist(),
                    self.pad_tag,
                    start_tag=self.start_tag,
                    end_tag=self.end_tag,
                )
            )

        y = []
        for _, tag_ids_array in self.validation_data.as_numpy_iterator():
            y.extend(
                [
                    convert_ids_to_tags(
                        self.idx2tag,
                        tag_ids,
                        self.pad_tag,
                        start_tag=self.start_tag,
                        end_tag=self.end_tag,
                    )
                    for tag_ids in tag_ids_array.tolist()
                ]
            )

        score_df = tagging_fscore(y, predictions, self.labels)
        logger.info(score_df)
        row = score_df[score_df["class"] == "avg"]
        for col in score_df.columns:
            if col == "support":
                continue
            logs[f"val_{col}"] = row[col].values.tolist()[0]