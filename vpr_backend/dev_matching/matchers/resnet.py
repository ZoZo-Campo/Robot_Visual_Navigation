from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .base import BaseMatcher


class ResNet18Matcher(BaseMatcher):
    name = "resnet18"

    def extract(self, paths: Sequence[Path]) -> np.ndarray:
        try:
            import torch
            from PIL import Image
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise RuntimeError("ResNet18 requires torch and torchvision; run scripts/setup_mac.sh") from exc
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval().to(self.device)
        transform = weights.transforms()
        rows = []
        with torch.inference_mode():
            for start in range(0, len(paths), self.batch_size):
                batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[start:start + self.batch_size]])
                rows.append(model(batch.to(self.device)).cpu().numpy())
        return np.concatenate(rows).astype(np.float32)
