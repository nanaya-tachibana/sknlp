from typing import Any, Dict, Sequence, Optional

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout
import pandas as pd

from sknlp.layers import (
    BertLayer,
    BertCharPreprocessingLayer,
)
from sknlp.data import BertTaggingDataset
from sknlp.module.text2vec import Bert2vec
from .deep_tagger import DeepTagger


class BertTagger(DeepTagger):
    def __init__(
        self,
        classes: Sequence[str],
        segmenter: Optional[str] = None,
        use_crf: bool = False,
        crf_learning_rate_multiplier: float = 1.0,
        embedding_size: int = 100,
        max_sequence_length: int = 120,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 256,
        fc_activation: str = "tanh",
        output_dropout: float = 0.5,
        text2vec: Optional[Bert2vec] = None,
        **kwargs
    ) -> None:
        super().__init__(
            classes,
            segmenter=segmenter,
            use_crf=use_crf,
            crf_learning_rate_multiplier=crf_learning_rate_multiplier,
            algorithm="bert",
            embedding_size=embedding_size,
            num_fc_layers=num_fc_layers,
            fc_hidden_size=fc_hidden_size,
            fc_activation=fc_activation,
            max_sequence_length=max_sequence_length,
            start_tag="[CLS]",
            end_tag="[SEP]",
            text2vec=text2vec,
            **kwargs
        )
        self.output_dropout = output_dropout
        self.inputs = [
            tf.keras.Input(shape=(), dtype=tf.string, name="text_input"),
            tf.keras.Input(shape=(None,), dtype=tf.int32, name="tag_id"),
        ]

    def create_dataset_from_df(
        self,
        df: pd.DataFrame,
        no_label: bool = False,
    ) -> BertTaggingDataset:
        return BertTaggingDataset(
            self.text2vec.vocab,
            self.classes,
            df=df,
            max_length=self.max_sequence_length,
            no_label=no_label,
            use_crf=self.use_crf,
        )

    def create_dataset_from_csv(
        self,
        filename: str,
        no_label: bool = False,
    ) -> BertTaggingDataset:
        return BertTaggingDataset(
            self.text2vec.vocab,
            self.classes,
            csv_file=filename,
            max_length=self.max_sequence_length,
            no_label=no_label,
            use_crf=self.use_crf,
        )

    def build_encode_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        texts, tag_ids = inputs
        token_ids = BertCharPreprocessingLayer(self.text2vec.vocab.sorted_tokens)(texts)
        embeddings, mask, _ = self.text2vec(
            [token_ids, K.zeros_like(token_ids, dtype=tf.int64)]
        )
        mask = tf.keras.layers.Lambda(
            lambda x: tf.cast(x, tf.int32), name="mask_layer"
        )(mask)
        if self.output_dropout:
            noise_shape = (None, 1, self.embedding_size)
            embeddings = Dropout(
                self.output_dropout,
                noise_shape=noise_shape,
                name="embedding_dropout",
            )(embeddings)
        return embeddings, mask, tag_ids

    @classmethod
    def _filter_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        config = super()._filter_config(config)
        config.pop("start_tag", None)
        config.pop("end_tag", None)
        return config

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "output_dropout": self.output_dropout}

    def get_custom_objects(self) -> Dict[str, Any]:
        return {
            **super().get_custom_objects(),
            "BertLayer": BertLayer,
            "BertCharPreprocessingLayer": BertCharPreprocessingLayer,
        }
