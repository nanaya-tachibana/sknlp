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
from sknlp.data import BertClassificationDataset
from sknlp.module.text2vec import Bert2vec
from .deep_classifier import DeepClassifier


class BertClassifier(DeepClassifier):
    def __init__(
        self,
        classes: Sequence[str],
        is_multilabel: bool = True,
        segmenter: Optional[str] = None,
        embedding_size: int = 100,
        use_batch_normalization: bool = True,
        max_sequence_length: int = 120,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 256,
        fc_activation: str = "tanh",
        fc_momentum: float = 0.9,
        fc_epsilon: float = 1e-5,
        output_dropout: float = 0.5,
        text2vec: Optional[Bert2vec] = None,
        **kwargs
    ) -> None:
        super().__init__(
            classes,
            is_multilabel=is_multilabel,
            segmenter=segmenter,
            algorithm="bert",
            embedding_size=embedding_size,
            use_batch_normalization=use_batch_normalization,
            max_sequence_length=max_sequence_length,
            text2vec=text2vec,
            **kwargs
        )
        self.num_fc_layers = num_fc_layers
        self.fc_hidden_size = fc_hidden_size
        self.fc_activation = fc_activation
        self.fc_momentum = fc_momentum
        self.fc_epsilon = fc_epsilon
        self.output_dropout = output_dropout
        self.inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")

    def create_dataset_from_df(
        self,
        df: pd.DataFrame,
        no_label: bool = False,
    ) -> BertClassificationDataset:
        return BertClassificationDataset(
            self.text2vec.vocab,
            self.classes,
            df=df,
            is_multilabel=self.is_multilabel,
            max_length=self.max_sequence_length,
            no_label=no_label,
        )

    def create_dataset_from_csv(
        self,
        filename: str,
        no_label: bool = False,
    ) -> BertClassificationDataset:
        return BertClassificationDataset(
            self.text2vec.vocab,
            self.classes,
            csv_file=filename,
            is_multilabel=self.is_multilabel,
            max_length=self.max_sequence_length,
            no_label=no_label,
        )

    def build_encode_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        preprocessing_layer = BertPreprocessingLayer(self.text2vec.vocab.sorted_tokens)
        token_ids = preprocessing_layer(inputs)
        _, _, cls = self.text2vec(
            [
                token_ids,
                K.zeros_like(token_ids, dtype=tf.int64),
            ]
        )
        if self.output_dropout:
            return Dropout(self.output_dropout, name="embedding_dropout")(cls)
        return cls

    def build_output_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        return MLPLayer(
            self.num_fc_layers,
            hidden_size=self.fc_hidden_size,
            output_size=self.num_classes,
            activation=self.fc_activation,
            batch_normalization=self.use_batch_normalization,
            momentum=self.fc_momentum,
            epsilon=self.fc_epsilon,
            name="mlp",
        )(inputs)

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "output_dropout": self.output_dropout}

    def get_custom_objects(self) -> Dict[str, Any]:
        return {
            **super().get_custom_objects(),
            "MLPLayer": MLPLayer,
            "BertLayer": BertLayer,
            "BertPreprocessingLayer": BertPreprocessingLayer,
        }
