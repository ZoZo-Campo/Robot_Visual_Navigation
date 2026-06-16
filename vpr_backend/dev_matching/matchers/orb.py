from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .base import BaseMatcher, MatcherCapabilities


class ORBMatcher(BaseMatcher):
    """Bag-of-bits ORB descriptor with pairwise geometric verification support."""

    name = "orb"
    capabilities = MatcherCapabilities(supports_batching=False, supports_pair_visualization=True)

    def extract(self, paths: Sequence[Path]) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("ORB requires the 'vision' extra: pip install -e '.[vision]'") from exc
        nfeatures = int(self.params.get("nfeatures", 1000))
        orb = cv2.ORB_create(nfeatures=nfeatures)
        rows = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Cannot read image: {path}")
            _, descriptors = orb.detectAndCompute(image, None)
            # A fixed 256-bin bit-frequency representation allows dense cached comparison.
            if descriptors is None:
                rows.append(np.zeros(256, dtype=np.float32))
            else:
                bits = np.unpackbits(descriptors, axis=1)
                rows.append(bits.mean(axis=0).astype(np.float32))
        return np.stack(rows)

    def verify_pair(self, query_path: Path, database_path: Path) -> dict[str, float | int]:
        import cv2

        orb = cv2.ORB_create(nfeatures=int(self.params.get("nfeatures", 1000)))
        left = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(database_path), cv2.IMREAD_GRAYSCALE)
        kp1, des1 = orb.detectAndCompute(left, None)
        kp2, des2 = orb.detectAndCompute(right, None)
        if des1 is None or des2 is None:
            return {"matches": 0, "inliers": 0, "inlier_ratio": 0.0}
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
        good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            return {"matches": len(good), "inliers": 0, "inlier_ratio": 0.0}
        src = np.float32([kp1[m.queryIdx].pt for m in good])
        dst = np.float32([kp2[m.trainIdx].pt for m in good])
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        return {"matches": len(good), "inliers": inliers, "inlier_ratio": inliers / len(good)}
