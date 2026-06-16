from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class MatcherCapabilities:
    fixed_size_descriptor: bool = True
    supports_batching: bool = True
    supports_pair_visualization: bool = False


class BaseMatcher(ABC):
    name: str
    capabilities = MatcherCapabilities()

    def __init__(self, device: str = "cpu", batch_size: int = 16, **params: Any):
        self.device = device
        self.batch_size = batch_size
        self.params = params

    @property
    def cache_signature(self) -> dict[str, Any]:
        return {"name": self.name, "device_independent_params": self.params}

    @abstractmethod
    def extract(self, paths: Sequence[Path]) -> np.ndarray:
        """Return one descriptor row per path."""

    def similarity_matrix(self, query: np.ndarray, database: np.ndarray) -> np.ndarray:
        query = self._normalize(query)
        database = self._normalize(database)
        return np.clip(query @ database.T, -1.0, 1.0)

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        values = np.asarray(array, dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, 1e-12)
