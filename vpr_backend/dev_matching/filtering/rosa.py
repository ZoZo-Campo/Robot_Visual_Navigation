from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dev_matching.config import FilteringConfig
from dev_matching.data import ImageCollection

if TYPE_CHECKING:
    from dev_matching.pipeline.geometry import GeometryDiagnostics


@dataclass(frozen=True)
class FilterDecision:
    original_index: int
    selected: bool
    reason: str
    representative_index: int | None
    route_position_m: float | None
    distance_from_last_kept_m: float | None
    similarity_to_last_kept: float | None


@dataclass(frozen=True)
class RosaFilterResult:
    method: str
    selected_indices: np.ndarray
    decisions: tuple[FilterDecision, ...]

    @property
    def rejected_indices(self) -> np.ndarray:
        return np.asarray(
            [decision.original_index for decision in self.decisions if not decision.selected],
            dtype=np.int64,
        )

    @property
    def counts_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for decision in self.decisions:
            counts[decision.reason] = counts.get(decision.reason, 0) + 1
        return counts


def _normalized(descriptors: np.ndarray) -> np.ndarray:
    values = np.asarray(descriptors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("ROSA descriptors must be a 2D matrix")
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _quality_key(index: int, collection: ImageCollection, geometry: GeometryDiagnostics) -> tuple[float, float, int]:
    record = collection.records[index]
    pano_distance = record.distance_to_pano_m if record.distance_to_pano_m is not None else float("inf")
    heading_error = geometry.heading_errors_deg[index]
    heading_error_value = float(heading_error) if not np.isnan(heading_error) else float("inf")
    return pano_distance, heading_error_value, index


def filter_rosa(
    collection: ImageCollection,
    descriptors: np.ndarray,
    geometry: GeometryDiagnostics,
    config: FilteringConfig,
) -> RosaFilterResult:
    """ROSA: Route-Oriented Similarity-Aware reference filtering."""
    if len(descriptors) != len(collection):
        raise ValueError("ROSA descriptor count does not match the Street View collection")
    values = _normalized(descriptors)
    count = len(collection)
    reasons = ["candidate"] * count
    representatives: list[int | None] = [None] * count
    distances: list[float | None] = [None] * count
    similarities: list[float | None] = [None] * count

    eligible = np.ones(count, dtype=bool)
    if config.enforce_dominant_direction and np.any(~geometry.primary_route_mask):
        eligible &= geometry.primary_route_mask
        for index in np.flatnonzero(~geometry.primary_route_mask):
            reasons[int(index)] = "wrong_direction"

    candidates = [int(index) for index in np.flatnonzero(eligible)]
    if config.deduplicate_panoramas:
        groups: dict[str, list[int]] = {}
        for index in candidates:
            pano_id = collection.records[index].pano_id
            key = f"pano:{pano_id}" if pano_id else f"image:{index}"
            groups.setdefault(key, []).append(index)
        representatives_by_group: list[int] = []
        for group in groups.values():
            representative = min(group, key=lambda index: _quality_key(index, collection, geometry))
            representatives_by_group.append(representative)
            for index in group:
                representatives[index] = representative
                if index != representative:
                    reasons[index] = "duplicate_panorama"
                    eligible[index] = False
        candidates = sorted(representatives_by_group)

    if not candidates:
        raise ValueError("ROSA rejected every Street View image")

    sequence_outliers: list[int] = []
    for previous, current, following in zip(candidates, candidates[1:], candidates[2:]):
        left_similarity = float(values[current] @ values[previous])
        right_similarity = float(values[current] @ values[following])
        bridge_similarity = float(values[previous] @ values[following])
        if (
            max(left_similarity, right_similarity) < config.sequence_outlier_neighbor_max
            and bridge_similarity >= config.sequence_outlier_bridge_min
        ):
            sequence_outliers.append(current)
            reasons[current] = "sequence_outlier"
            representatives[current] = previous
            eligible[current] = False
    if sequence_outliers:
        rejected = set(sequence_outliers)
        candidates = [index for index in candidates if index not in rejected]

    positions = geometry.route_positions_m
    kept: list[int] = []
    for candidate_position, index in enumerate(candidates):
        if not eligible[index]:
            continue
        if not kept:
            kept.append(index)
            reasons[index] = "kept_endpoint"
            continue
        previous = kept[-1]
        if positions is not None:
            gap = float(positions[index] - positions[previous])
        else:
            gap = float(index - previous)
        similarity = float(values[index] @ values[previous])
        distances[index] = gap
        similarities[index] = similarity
        is_last = candidate_position == len(candidates) - 1
        if is_last:
            kept.append(index)
            reasons[index] = "kept_endpoint"
        elif gap < config.min_spacing_m:
            reasons[index] = "too_close"
            representatives[index] = previous
        elif similarity >= config.visual_similarity_threshold and gap < config.max_spacing_m:
            reasons[index] = "visual_redundancy"
            representatives[index] = previous
        else:
            kept.append(index)
            reasons[index] = "kept"

    kept_set = set(kept)
    decisions = tuple(
        FilterDecision(
            original_index=index,
            selected=index in kept_set,
            reason=reasons[index],
            representative_index=representatives[index],
            route_position_m=float(positions[index]) if positions is not None else None,
            distance_from_last_kept_m=distances[index],
            similarity_to_last_kept=similarities[index],
        )
        for index in range(count)
    )
    return RosaFilterResult(config.method, np.asarray(kept, dtype=np.int64), decisions)


def identity_filter(collection: ImageCollection, method: str = "disabled") -> RosaFilterResult:
    decisions = tuple(
        FilterDecision(index, True, "kept", index, None, None, None)
        for index in range(len(collection))
    )
    return RosaFilterResult(method, np.arange(len(collection), dtype=np.int64), decisions)
