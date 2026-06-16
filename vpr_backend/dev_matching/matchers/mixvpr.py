from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .base import BaseMatcher, MatcherCapabilities


class MixVPRMatcher(BaseMatcher):
    """Official ResNet50 + MixVPR 4096-D global image descriptor."""

    name = "mixvpr"
    capabilities = MatcherCapabilities(supports_pair_visualization=True)
    _MODEL_CACHE: dict[tuple[str, str], Any] = {}

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 16,
        checkpoint: str | Path | None = None,
        **params: Any,
    ):
        super().__init__(device=device, batch_size=batch_size, checkpoint=checkpoint, **params)
        if checkpoint is None:
            raise ValueError("MixVPR requires params.checkpoint in the YAML configuration")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"MixVPR checkpoint not found: {self.checkpoint}")
        self._model = None

    @property
    def cache_signature(self) -> dict[str, Any]:
        digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        return {
            "name": self.name,
            "architecture": "resnet50_layer3_mixvpr_4096",
            "input_size": 320,
            "checkpoint_sha256": digest,
        }

    def _load_model(self):
        if self._model is not None:
            return self._model
        model_key = (str(self.checkpoint), self.device)
        if model_key in self._MODEL_CACHE:
            self._model = self._MODEL_CACHE[model_key]
            return self._model
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("MixVPR requires: pip install -e '.[torch]'") from exc

        model = _build_official_model(torch)
        try:
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(self.checkpoint, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise ValueError("The MixVPR checkpoint does not contain a state dictionary")

        state = {}
        for key, value in payload.items():
            clean_key = key
            for prefix in ("module.", "model."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            state[clean_key] = value
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                "Checkpoint incompatible with official MixVPR ResNet50 4096-D model. "
                f"Missing keys: {missing[:5]}; unexpected keys: {unexpected[:5]}"
            )
        self._model = model.eval().to(self.device)
        self._MODEL_CACHE[model_key] = self._model
        return self._model

    def extract(self, paths: Sequence[Path]) -> np.ndarray:
        if not paths:
            return np.empty((0, 4096), dtype=np.float32)
        try:
            import torch
            from PIL import Image
            from torchvision.transforms import v2
        except ImportError as exc:
            raise RuntimeError("MixVPR requires: pip install -e '.[torch]'") from exc

        model = self._load_model()
        transform = v2.Compose(
            [
                v2.Resize((320, 320), interpolation=v2.InterpolationMode.BICUBIC),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        rows = []
        with torch.inference_mode():
            for start in range(0, len(paths), self.batch_size):
                images = [Image.open(path).convert("RGB") for path in paths[start:start + self.batch_size]]
                batch = torch.stack([transform(image) for image in images]).to(self.device)
                rows.append(model(batch).cpu().numpy())
        return np.concatenate(rows).astype(np.float32, copy=False)


def _build_official_model(torch):
    """Build the architecture used by the official 4096-D pretrained checkpoint."""
    from torchvision.models import resnet50

    nn = torch.nn

    class FeatureMixerLayer(nn.Module):
        def __init__(self, hw: int):
            super().__init__()
            self.mix = nn.Sequential(
                nn.LayerNorm(hw),
                nn.Linear(hw, hw),
                nn.ReLU(),
                nn.Linear(hw, hw),
            )

        def forward(self, x):
            return x + self.mix(x)

    class MixVPR(nn.Module):
        def __init__(self):
            super().__init__()
            self.mix = nn.Sequential(*(FeatureMixerLayer(400) for _ in range(4)))
            self.channel_proj = nn.Linear(1024, 1024)
            self.row_proj = nn.Linear(400, 4)

        def forward(self, x):
            x = x.flatten(2)
            x = self.mix(x)
            x = x.permute(0, 2, 1)
            x = self.channel_proj(x)
            x = x.permute(0, 2, 1)
            x = self.row_proj(x)
            x = x.flatten(1)
            return nn.functional.normalize(x, p=2, dim=1)

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = resnet50(weights=None)
            self.model.avgpool = None
            self.model.fc = None
            self.model.layer4 = None

        def forward(self, x):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            return self.model.layer3(x)

    class VPRModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
            self.aggregator = MixVPR()

        def forward(self, x):
            return self.aggregator(self.backbone(x))

    return VPRModel()
