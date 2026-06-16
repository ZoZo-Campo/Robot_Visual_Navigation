from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

from .base import BaseMatcher


class SpatialPyramidMatcher(BaseMatcher):
    """Multi-scale HSV and oriented-gradient descriptor requiring no model download."""

    name = "spatial_pyramid"

    def extract(self, paths: Sequence[Path]) -> np.ndarray:
        size = int(self.params.get("image_size", 256))
        levels = tuple(int(level) for level in self.params.get("levels", (1, 2, 4)))
        return np.stack([self._describe(path, size, levels) for path in paths]).astype(np.float32)

    @staticmethod
    def _describe(path: Path, size: int, levels: tuple[int, ...]) -> np.ndarray:
        with Image.open(path) as source:
            rgb_image = ImageOps.fit(ImageOps.exif_transpose(source).convert("RGB"), (size, size))
            hsv = np.asarray(rgb_image.convert("HSV"), dtype=np.uint8)
            gray = np.asarray(rgb_image.convert("L"), dtype=np.float32) / 255.0

        gy, gx = np.gradient(gray)
        magnitude = np.hypot(gx, gy)
        orientation = (np.arctan2(gy, gx) + np.pi) % np.pi
        pieces: list[np.ndarray] = []
        for level in levels:
            edges = np.linspace(0, size, level + 1, dtype=int)
            for row in range(level):
                for column in range(level):
                    y0, y1 = edges[row], edges[row + 1]
                    x0, x1 = edges[column], edges[column + 1]
                    cell = hsv[y0:y1, x0:x1]
                    color = [
                        np.histogram(cell[..., 0], bins=16, range=(0, 256))[0],
                        np.histogram(cell[..., 1], bins=8, range=(0, 256))[0],
                        np.histogram(cell[..., 2], bins=8, range=(0, 256))[0],
                    ]
                    color_vector = np.concatenate(color).astype(np.float32)
                    color_vector /= max(float(color_vector.sum()), 1.0)

                    angles = orientation[y0:y1, x0:x1].ravel()
                    weights = magnitude[y0:y1, x0:x1].ravel()
                    gradient = np.histogram(angles, bins=9, range=(0, np.pi), weights=weights)[0].astype(np.float32)
                    gradient /= max(float(gradient.sum()), 1e-6)
                    pieces.extend((color_vector, gradient))
        return np.concatenate(pieces)
