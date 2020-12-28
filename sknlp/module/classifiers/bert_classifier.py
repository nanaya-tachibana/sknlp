from typing import Any, Dict, Sequence, Optional

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout
import pandas as pd

from sknlp.layers import (
    MLPLayer,
    BertLayer,
    BertPreprocessingLayer,
)
from sknlp.vocab import Vocab
from sknlp.data import BertClassificationDataset
from ..text2vec import Bert2vec
from .deep_classifier import DeepClassifier


class BertClassifier(DeepClassifier):
    def __init__(
        self,
        classes: Sequence[str],
        is_multilabel: bool = True,
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
            is_multilabel=is_multilabel,
            segmenter=segmenter,
            algorithm="bert_classifier",
            embedding_size=embedding_size,
            max_sequence_length=max_sequence_length,
            text2vec=text2vec,
            **kwargs
        )
        # if self._text2vec is not None:
        #     self.preprocessing_layer = BertPreprocessingLayer(
        #         self._text2vec.vocab.sorted_tokens
        #     )
        
        self.num_fc_layers = num_fc_layers
        self.fc_hidden_size = fc_hidden_size
        self.inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")

        # self.inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")
        self._output_dropout = output_dropout

    def create_dataset_from_df(
        self, df: pd.DataFrame, vocab: Vocab, segmenter: str, labels: Sequence[str]
    ) -> BertClassificationDataset:
        return BertClassificationDataset(
            vocab,
            list(labels),
            df=df,
            is_multilabel=self._is_multilabel,
            max_length=self._max_sequence_length
        )

    def get_inputs(self) -> tf.Tensor:
        return self.inputs

    def get_outputs(self) -> tf.Tensor:
        return self.build_output_layer(self.build_encode_layer(self.get_inputs()))

    def build_encode_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        preprocessing_layer = BertPreprocessingLayer(
            self._text2vec.vocab.sorted_tokens
        )
        token_ids = preprocessing_layer(inputs)
        _, _, cls = self._text2vec(
            [
                token_ids,
                K.zeros_like(token_ids, dtype=tf.int64),
            ]
        )
        if self._output_dropout:
            return Dropout(self._output_dropout)(cls)
        return cls

    def build_output_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        return MLPLayer(
            self.num_fc_layers,
            hidden_size=self.fc_hidden_size,
            output_size=self.num_classes,
            name="mlp",
        )(inputs)

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "output_dropout": self._output_dropout}

    def get_custom_objects(self) -> Dict[str, Any]:
        return {
            **super().get_custom_objects(),
            "MLPLayer": MLPLayer,
            "BertLayer": BertLayer,
            "BertPreprocessingLayer": BertPreprocessingLayer,
        }
