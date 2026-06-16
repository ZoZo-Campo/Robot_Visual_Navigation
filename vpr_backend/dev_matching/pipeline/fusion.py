from __future__ import annotations

import numpy as np


def _rank_normalize(matrix: np.ndarray) -> np.ndarray:
    """Convert each query row to [0, 1] ranks, making matcher scales comparable."""
    if matrix.shape[1] == 1:
        return np.ones_like(matrix, dtype=np.float32)
    order = np.argsort(matrix, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(matrix.shape[0])[:, None]
    ranks[rows, order] = np.arange(matrix.shape[1], dtype=np.float32)
    return ranks / (matrix.shape[1] - 1)


def fuse_matrices(matrices: dict[str, np.ndarray], weights: dict[str, float], mode: str = "rank") -> np.ndarray:
    if not matrices:
        raise ValueError("No similarity matrices to fuse")
    selected = [(name, matrix, float(weights[name])) for name, matrix in matrices.items() if weights.get(name, 0) > 0]
    if not selected:
        raise ValueError("At least one fusion weight must be positive")
    transform = _rank_normalize if mode == "rank" and len(selected) > 1 else lambda value: value
    total = sum(weight for _, _, weight in selected)
    return sum(weight * transform(matrix) for _, matrix, weight in selected) / total
