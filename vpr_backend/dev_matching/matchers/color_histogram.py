from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .base import BaseMatcher


class ColorHistogramMatcher(BaseMatcher):
    """Fast dependency-light baseline useful for validation and CI."""

    name = "color_histogram"

    def extract(self, paths: Sequence[Path]) -> np.ndarray:
        bins = int(self.params.get("bins", 32))
        descriptors = []
        for path in paths:
            image = np.asarray(Image.open(path).convert("RGB"))
            channels = [np.histogram(image[..., c], bins=bins, range=(0, 256))[0] for c in range(3)]
            descriptors.append(np.concatenate(channels).astype(np.float32))
        return np.stack(descriptors)
