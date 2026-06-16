from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_k: dict[int, float]
    mrr: float
    top1_accuracy: float
    median_rank: float


def evaluate_retrieval(
    similarity: np.ndarray,
    ground_truth: Iterable[Iterable[int]],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> RetrievalMetrics:
    truth = [set(map(int, matches)) for matches in ground_truth]
    if len(truth) != similarity.shape[0] or any(not item for item in truth):
        raise ValueError("Ground truth must provide at least one reference index per query")
    order = np.argsort(-similarity, axis=1)
    ranks = []
    for i, valid in enumerate(truth):
        rank = min(int(np.flatnonzero(np.isin(order[i], list(valid)))[0]) + 1, similarity.shape[1])
        ranks.append(rank)
    ranks_array = np.asarray(ranks)
    return RetrievalMetrics(
        recall_at_k={k: float(np.mean(ranks_array <= k)) for k in ks},
        mrr=float(np.mean(1.0 / ranks_array)),
        top1_accuracy=float(np.mean(ranks_array == 1)),
        median_rank=float(np.median(ranks_array)),
    )


def localization_errors(predicted: Iterable[int], expected: Iterable[int], spacing_meters: float = 1.0) -> np.ndarray:
    return np.abs(np.asarray(list(predicted)) - np.asarray(list(expected))) * spacing_meters
