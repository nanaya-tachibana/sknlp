from __future__ import annotations
from typing import Optional, Any, Union, Sequence
import tensorflow as tf

import pandas as pd

from sknlp.data import GenerationDataset
from sknlp.module.supervised_model import SupervisedNLPModel
from sknlp.module.text2vec import Text2vec
from sknlp.utils.generation import corpus_bleu


class DeepGenerator(SupervisedNLPModel):
    dataset_class = GenerationDataset

    def __init__(
        self,
        max_sequence_length: Optional[int] = None,
        beam_width: int = 3,
        text2vec: Optional[Text2vec] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            [],
            max_sequence_length=max_sequence_length,
            text2vec=text2vec,
            task="generation",
            **kwargs,
        )
        self._beam_width = beam_width

    @property
    def beam_width(self) -> int:
        return self._beam_width

    def get_loss(self, *args, **kwargs) -> tf.keras.losses.Loss:
        return None

    def get_monitor(self) -> str:
        return "val_lm_accuracy"

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "beam_width": self.beam_width}

    def predict(
        self,
        X: Sequence[str] = None,
        *,
        dataset: GenerationDataset = None,
        thresholds: Union[float, list[float], None] = None,
        batch_size: int = 128,
    ) -> list[str]:
        predictions = super().predict(
            X=X, dataset=dataset, thresholds=thresholds, batch_size=batch_size
        )
        vocab = self.text2vec.vocab
        reversed_tokens = {
            vocab.pad,
            vocab.unk,
            vocab.bos,
            vocab.eos,
        }
        generations: list[str] = []
        for tokens in vocab.idx2token(predictions.tolist()):
            translated_tokens = []
            for token in tokens:
                if token == vocab.bos:
                    break
                if token in reversed_tokens:
                    continue
                if token.startswith("##"):
                    token = token[2:]
                translated_tokens.append(token)
            generations.append("".join(translated_tokens))
        return generations

    def score(
        self,
        X: Sequence[str] = None,
        y: Sequence[str] = None,
        *,
        dataset: GenerationDataset = None,
        thresholds: Union[float, list[float], None] = None,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        dataset = self.prepare_dataset(X=X, y=y, dataset=dataset)
        generations = self.predict(
            dataset=dataset, thresholds=thresholds, batch_size=batch_size
        )
        bleu = corpus_bleu([[yi] for yi in dataset.y], generations)
        return pd.DataFrame([("bleu", bleu)], columns=["score", "value"])

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DeepGenerator":
        config.pop("task", None)
        config.pop("algorithm", None)
        config.pop("classes", None)
        return super().from_config(config)