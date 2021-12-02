from __future__ import annotations
from typing import Sequence, Any, Optional
from itertools import cycle


import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
import igraph

from sknlp.data import RetrievalDataset, RetrievalEvaluationDataset
from sknlp.module.supervised_model import SupervisedNLPModel
from sknlp.module.text2vec import Text2vec


def build_graph_from_file(filename: str) -> igraph.Graph:
    df = pd.read_csv(filename, sep="\t", quoting=3, escapechar="\\")
    texts = list(set(df.iloc[:, 0]) | set(df.iloc[:, 1]))
    g = igraph.Graph(n=len(texts), vertex_attrs={"name": texts})
    edges = []
    edge_types = []
    for _, row in df.iterrows():
        edges.append((g.vs.find(row[0]), g.vs.find(row[1])))
        edge_types.append(str(int(row[2])))
    g.add_edges(edges)
    g.es["type"] = edge_types
    return g


def neighbors_lower_than_kth_order(
    g: igraph.Graph, v: int, max_order: int
) -> tuple[list[set[int]], list[set[int]]]:
    kth_order_positive_neighbors: list[set[int]] = [{v}]
    kth_order_negative_neighbors: list[set[int]] = [set()]
    for k in range(max_order + 1):
        positive_neighbors = set()
        negative_neighbors = set()
        for vi in kth_order_positive_neighbors[k]:
            for ui in g.neighbors(vi):
                eid = g.get_eid(ui, vi)
                edge_type = g.es[eid]["type"]
                if edge_type == "0":
                    negative_neighbors.add(ui)
                else:
                    positive_neighbors.add(ui)
        kth_order_positive_neighbors.append(
            positive_neighbors - kth_order_positive_neighbors[-1]
        )
        kth_order_negative_neighbors.append(
            negative_neighbors - kth_order_negative_neighbors[-1]
        )
    return kth_order_positive_neighbors[1:], kth_order_negative_neighbors[1:]


def match_positive_negative_set_size(
    positive_set_list: list[set[int]], negative_set_list: list[set[int]]
) -> tuple[list[int], list[int]]:
    positives = list(positive_set_list[0])
    negatives = list(negative_set_list[0])
    if len(positives) < len(negatives):
        i = 1
        while i < len(positive_set_list) and len(positives) < len(negatives):
            positives += list(positive_set_list[i])
            i += 1
    elif len(positives) > len(negatives):
        i = 1
        while i < len(negative_set_list) and len(positives) > len(negatives):
            negatives += list(negative_set_list[i])
            i += 1
    return positives, negatives


class DeepRetriever(SupervisedNLPModel):
    dataset_class = RetrievalDataset
    evaluation_dataset_class = RetrievalEvaluationDataset

    def __init__(
        self,
        classes: Sequence[int] = (0, 1),
        max_sequence_length: Optional[int] = None,
        has_negative: bool = False,
        projection_size: Optional[int] = None,
        temperature: float = 0.05,
        text2vec: Optional[Text2vec] = None,
        loss: Optional[str] = None,
        loss_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        classes = list(classes)
        super().__init__(
            classes,
            max_sequence_length=max_sequence_length,
            text2vec=text2vec,
            task="retrieval",
            **kwargs,
        )
        self._loss = loss
        self._loss_kwargs = loss_kwargs
        self.projection_size = projection_size
        self.temperature = temperature
        self.has_negative = has_negative

    def get_loss(self, *args, **kwargs) -> list[tf.keras.losses.Loss]:
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def get_metrics(self, *args, **kwargs) -> list[tf.keras.metrics.Metric]:
        return []

    def get_monitor(cls) -> str:
        return None

    def build_intermediate_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        pooling_layer = tf.keras.layers.Lambda(lambda x: x, name="pooling")
        if self.projection_size is not None:
            pooling_layer = tf.keras.layers.Dense(self.projection_size, name="pooling")
        return pooling_layer(inputs)

    def build_output_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        num_inputs = 2 + self.has_negative
        normalized = inputs / tf.linalg.norm(inputs, axis=-1, keepdims=True)
        reshaped = tf.reshape(normalized, (-1, num_inputs, tf.shape(inputs)[-1]))
        input_list: list[tf.Tensor] = tf.split(reshaped, num_inputs, axis=1)

        input = tf.squeeze(input_list[0], axis=1)
        positive = tf.squeeze(input_list[1], axis=1)
        cos_sim = tf.matmul(input, positive, transpose_b=True)
        if num_inputs == 3:
            negative = tf.squeeze(input_list[2], axis=1)
            cos_sim = tf.concat(
                [cos_sim, tf.matmul(input, negative, transpose_b=True)], -1
            )
        return cos_sim / self.temperature

    def build_inference_model(self) -> tf.keras.Model:
        self._inference_model = tf.keras.Model(
            inputs=self._model.inputs,
            outputs=self._model.get_layer("pooling").output,
        )

    def predict(
        self,
        X: Sequence[tuple[str, str]] = None,
        *,
        dataset: RetrievalDataset = None,
        batch_size: int = 128,
    ) -> list[float]:
        predictions = super().predict(X=X, dataset=dataset, batch_size=batch_size)
        normalized = predictions / np.linalg.norm(predictions, axis=-1, keepdims=True)
        return (normalized[::2, :] * normalized[1::2, :]).sum(axis=-1).tolist()

    def score(
        self,
        X: Sequence[tuple[str, str]] = None,
        y: Sequence[int] = None,
        *,
        dataset: RetrievalDataset = None,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        dataset = self.prepare_dataset(X, y, dataset, evaluation=True)
        predictions = self.predict(dataset=dataset, batch_size=batch_size)
        spearman = stats.spearmanr(dataset.y, predictions).correlation
        return pd.DataFrame([("spearman", spearman)], columns=["score", "value"])

    def create_dataset_from_01_label_dataset(
        self,
        filename: str,
        max_order: int = 3,
        evaluation: bool = False,
    ) -> RetrievalDataset:
        if evaluation:
            return self.create_dataset_from_csv(
                filename, has_label=True, evaluation=True
            )

        g = build_graph_from_file(filename)
        X: list[str] = []
        y: list[tuple[str, str]] = []
        for v in range(g.vcount()):
            positive_neighbors, negative_neighbors = match_positive_negative_set_size(
                *neighbors_lower_than_kth_order(g, v, max_order)
            )
            if not positive_neighbors:
                positive_neighbors = [v]
            if not negative_neighbors:
                negative_neighbors = [-1]
            p_size = len(positive_neighbors)
            n_size = len(negative_neighbors)
            if p_size > n_size:
                negative_neighbors = cycle(negative_neighbors)
            elif p_size < n_size:
                positive_neighbors = cycle(positive_neighbors)
            for pos, neg in zip(positive_neighbors, negative_neighbors):
                X.append(g.vs[v]["name"])
                y.append((g.vs[pos]["name"], "" if neg == -1 else g.vs[neg]["name"]))

        return self.dataset_class(
            self.vocab,
            self.classes,
            segmenter=self.segmenter,
            X=X,
            y=y,
            has_label=True,
            max_length=self.max_sequence_length,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "projection_size": self.projection_size,
            "temperature": self.temperature,
            "has_negative": self.has_negative,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DeepRetriever":
        config.pop("task", None)
        config.pop("algorithm", None)
        return super().from_config(config)