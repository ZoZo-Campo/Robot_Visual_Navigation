from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatabaseQualityReport:
    neighbor_similarity: np.ndarray
    robust_z_score: np.ndarray
    suggested_outliers: np.ndarray


def analyze_database(descriptors: np.ndarray, z_threshold: float = -3.5) -> DatabaseQualityReport:
    """Flag sequence-inconsistent references without silently removing them."""
    values = np.asarray(descriptors, dtype=np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    n = len(values)
    consistency = np.ones(n, dtype=np.float32)
    if n > 1:
        adjacent = np.sum(values[:-1] * values[1:], axis=1)
        consistency[0], consistency[-1] = adjacent[0], adjacent[-1]
        if n > 2:
            consistency[1:-1] = (adjacent[:-1] + adjacent[1:]) / 2
    median = np.median(consistency)
    mad = np.median(np.abs(consistency - median))
    z = 0.6745 * (consistency - median) / max(float(mad), 1e-6)
    return DatabaseQualityReport(consistency, z, np.flatnonzero(z < z_threshold))
