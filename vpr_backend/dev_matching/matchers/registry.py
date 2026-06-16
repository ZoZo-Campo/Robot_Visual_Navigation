from __future__ import annotations

from typing import Any

from .base import BaseMatcher
from .color_histogram import ColorHistogramMatcher
from .dinov2 import DINOv2Matcher
from .mixvpr import MixVPRMatcher
from .orb import ORBMatcher
from .resnet import ResNet18Matcher
from .spatial_pyramid import SpatialPyramidMatcher

_MATCHERS: dict[str, type[BaseMatcher]] = {
    cls.name: cls
    for cls in (ColorHistogramMatcher, SpatialPyramidMatcher, ORBMatcher, ResNet18Matcher, DINOv2Matcher, MixVPRMatcher)
}


def register_matcher(name: str, matcher: type[BaseMatcher]) -> None:
    if name in _MATCHERS:
        raise ValueError(f"Matcher already registered: {name}")
    _MATCHERS[name] = matcher


def build_matcher(name: str, **kwargs: Any) -> BaseMatcher:
    try:
        return _MATCHERS[name](**kwargs)
    except KeyError as exc:
        raise KeyError(f"Unknown matcher '{name}'. Available: {', '.join(sorted(_MATCHERS))}") from exc


def registered_matchers() -> tuple[str, ...]:
    return tuple(sorted(_MATCHERS))
