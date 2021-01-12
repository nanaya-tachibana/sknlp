from typing import Any, Dict, Sequence, Optional

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout
import pandas as pd

from sknlp.layers import (
    MLPLayer,
    BertLayer,
    BertCharPreprocessingLayer,
    CrfLossLayer,
    CrfDecodeLayer,
)
from sknlp.vocab import Vocab
from sknlp.data import BertTaggingDataset
from sknlp.module.text2vec import Bert2vec
from .deep_tagger import DeepTagger


class BertTagger(DeepTagger):
    def __init__(
        self,
        classes: Sequence[str],
        segmenter: Optional[str] = None,
        embedding_size: int = 100,
        max_sequence_length: int = 120,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 256,
        output_dropout: float = 0.5,
        text2vec: Optional[Bert2vec] = None,
        **kwargs
    ) -> None:
        super().__init__(
            classes,
            segmenter=segmenter,
            algorithm="bert",
            embedding_size=embedding_size,
            max_sequence_length=max_sequence_length,
            start_tag="[CLS]",
            end_tag="[SEP]",
            text2vec=text2vec,
            **kwargs
        )
        self.num_fc_layers = num_fc_layers
        self.fc_hidden_size = fc_hidden_size
        self.output_dropout = output_dropout
        self.inputs = [
            tf.keras.Input(shape=(), dtype=tf.string, name="text_input"),
            tf.keras.Input(shape=(None,), dtype=tf.int32, name="tag_id"),
        ]

    def create_dataset_from_df(
        self, df: pd.DataFrame, vocab: Vocab, segmenter: str, labels: Sequence[str]
    ) -> BertTaggingDataset:
        return BertTaggingDataset(
            vocab,
            list(labels),
            df=df,
            max_length=self.max_sequence_length,
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

    def build_output_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        embeddings, mask, tag_ids = inputs
        emissions = MLPLayer(
            self.num_fc_layers,
            hidden_size=self.fc_hidden_size,
            output_size=self.num_classes,
            name="mlp",
        )(embeddings)
        return CrfLossLayer(self.num_classes)([emissions, tag_ids], mask)

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
            "MLPLayer": MLPLayer,
            "BertLayer": BertLayer,
            "BertCharPreprocessingLayer": BertCharPreprocessingLayer,
            "CrfLossLayer": CrfLossLayer,
            "CrfDecodeLayer": CrfDecodeLayer,
        }
