from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .base import BaseMatcher, MatcherCapabilities


class DINOv2Matcher(BaseMatcher):
    name = "dinov2"
    capabilities = MatcherCapabilities(supports_pair_visualization=True)

    def extract(self, paths: Sequence[Path]) -> np.ndarray:
        raise RuntimeError(
            "DINOv2 is an optional provider adapter. Install torch and supply a locally available "
            "DINOv2 model implementation; see docs/EXTENDING.md. Runtime downloads are intentionally disabled."
        )
