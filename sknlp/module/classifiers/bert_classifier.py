from typing import Sequence, List, Dict, Any, Optional, Union

from tensorflow.keras.layers import Dropout
import pandas as pd

from sknlp.layers import MLPLayer
from sknlp.vocab import Vocab
from sknlp.data import BertClassificationDataset
from ..text2vec import Bert2vec
from .deep_classifier import DeepClassifier


class BertClassifier(DeepClassifier):

    def __init__(
        self,
        classes: Sequence[str],
        is_multilabel: bool = True,
        segmenter: str = None,
        embedding_size: int = 100,
        max_sequence_length: int = 512,
        num_fc_layers: int = 2,
        fc_hidden_size: int = 256,
        output_dropout: float = 0.5,
        text2vec: Optional[Bert2vec] = None,
        **kwargs
    ):
        super().__init__(classes,
                         is_multilabel=is_multilabel,
                         segmenter=segmenter,
                         algorithm="bert_classifier",
                         embedding_size=embedding_size,
                         max_sequence_length=max_sequence_length,
                         text2vec=text2vec,
                         **kwargs)
        self.mlp_layer = MLPLayer(num_fc_layers,
                                  hidden_size=fc_hidden_size,
                                  output_size=self.num_classes,
                                  name="mlp")
        self._output_dropout = output_dropout

    def create_dataset_from_df(
        self,
        df: pd.DataFrame,
        vocab: Vocab,
        segmenter: str,
        labels: Sequence[str]
    ) -> BertClassificationDataset:
        return BertClassificationDataset(
            vocab,
            list(labels),
            df=df,
            is_multilabel=self._is_multilabel,
            max_length=self._max_sequence_length,
            text_segmenter=segmenter
        )

    def build_encode_layer(self, inputs):
        _, cls = inputs
        if self._output_dropout:
            return Dropout(self._output_dropout)(cls)
        return cls

    def build_output_layer(self, inputs):
        return self.mlp_layer(inputs)

    def get_config(self):
        return {**super().get_config(), "output_dropout": self._output_dropout}

    @property
    def output_names(self) -> List[str]:
        return ["mlp"]

    @property
    def output_types(self) -> List[str]:
        return ["float"]

    @property
    def output_shapes(self) -> List[List[int]]:
        return [[-1, self.num_classes]]

    def get_custom_objects(self):
        return {**super().get_custom_objects(), 'MLPLayer': MLPLayer}
