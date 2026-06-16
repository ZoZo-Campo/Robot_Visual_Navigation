from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from dev_matching.data import ImageCollection


class DescriptorCache:
    """Disk cache for fixed-size descriptor matrices."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fingerprint(collection: ImageCollection, signature: dict) -> str:
        files = [
            {"path": str(r.path), "size": r.path.stat().st_size, "mtime_ns": r.path.stat().st_mtime_ns}
            for r in collection.records
        ]
        payload = json.dumps({"files": files, "signature": signature}, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:20]

    def path_for(self, collection: ImageCollection, method: str, signature: dict) -> Path:
        key = self._fingerprint(collection, signature)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in collection.name)
        return self.root / method / f"{safe_name}-{key}.npz"

    def get_or_compute(
        self,
        collection: ImageCollection,
        method: str,
        signature: dict,
        compute: Callable[[], np.ndarray],
    ) -> np.ndarray:
        path = self.path_for(collection, method, signature)
        if path.exists():
            with np.load(path, allow_pickle=False) as payload:
                return payload["descriptors"]
        descriptors = np.asarray(compute())
        if descriptors.shape[0] != len(collection):
            raise ValueError("Descriptor count does not match image count")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, descriptors=descriptors)
        temporary.replace(path)
        return descriptors
