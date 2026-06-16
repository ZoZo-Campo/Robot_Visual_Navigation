from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, radians, sin

import numpy as np

from dev_matching.config import GeometryConfig
from dev_matching.data import ImageCollection
from dev_matching.evaluation import haversine_meters


@dataclass(frozen=True)
class GeometryDiagnostics:
    route_positions_m: np.ndarray | None
    route_bearings_deg: np.ndarray
    heading_errors_deg: np.ndarray
    heading_alignment: np.ndarray
    pano_distance_quality: np.ndarray
    prior: np.ndarray
    route_segments: np.ndarray
    primary_route_mask: np.ndarray
    available: bool


def _bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_lon = radians(lon2 - lon1)
    y = sin(delta_lon) * cos(phi2)
    x = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(delta_lon)
    return (degrees(atan2(y, x)) + 360.0) % 360.0


def _angle_error(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def analyze_geometry(collection: ImageCollection, config: GeometryConfig) -> GeometryDiagnostics:
    count = len(collection)
    positions = None
    if all(record.has_position for record in collection.records):
        positions = np.zeros(count, dtype=np.float32)
        for index in range(1, count):
            previous = collection.records[index - 1]
            current = collection.records[index]
            positions[index] = positions[index - 1] + haversine_meters(
                previous.latitude, previous.longitude, current.latitude, current.longitude
            )

    bearings = np.full(count, np.nan, dtype=np.float32)
    if count > 1:
        for index, record in enumerate(collection.records):
            neighbor_index = index + 1 if index < count - 1 else index - 1
            neighbor = collection.records[neighbor_index]
            if record.has_position and neighbor.has_position:
                if index < count - 1:
                    bearings[index] = _bearing_degrees(
                        record.latitude, record.longitude, neighbor.latitude, neighbor.longitude
                    )
                else:
                    bearings[index] = _bearing_degrees(
                        neighbor.latitude, neighbor.longitude, record.latitude, record.longitude
                    )

    heading_errors = np.full(count, np.nan, dtype=np.float32)
    heading_alignment = np.ones(count, dtype=np.float32)
    distance_quality = np.ones(count, dtype=np.float32)
    has_heading = False
    has_distance = False
    for index, record in enumerate(collection.records):
        if record.heading is not None and not np.isnan(bearings[index]):
            error = _angle_error(record.heading, float(bearings[index]))
            heading_errors[index] = error
            heading_alignment[index] = np.exp(-0.5 * (error / max(config.heading_sigma_degrees, 1e-6)) ** 2)
            has_heading = True
        if record.distance_to_pano_m is not None:
            distance_quality[index] = np.exp(-record.distance_to_pano_m / max(config.pano_distance_scale_m, 1e-6))
            has_distance = True

    route_segments = np.zeros(count, dtype=np.int32)
    for index in range(1, count):
        previous_heading = collection.records[index - 1].heading
        current_heading = collection.records[index].heading
        if (
            previous_heading is not None
            and current_heading is not None
            and _angle_error(previous_heading, current_heading) > config.route_heading_break_degrees
        ):
            route_segments[index:] += 1
    segment_counts = np.bincount(route_segments) if count else np.array([], dtype=int)
    primary_segment = int(np.argmax(segment_counts)) if segment_counts.size else 0
    primary_route_mask = route_segments == primary_segment

    components = []
    if has_heading:
        components.append((0.8, heading_alignment))
    if has_distance:
        components.append((0.2, distance_quality))
    if components:
        total = sum(weight for weight, _ in components)
        prior = sum(weight * values for weight, values in components) / total
    else:
        prior = np.ones(count, dtype=np.float32)
    return GeometryDiagnostics(
        positions,
        bearings,
        heading_errors,
        heading_alignment,
        distance_quality,
        np.asarray(prior, dtype=np.float32),
        route_segments,
        primary_route_mask,
        bool(components),
    )


def apply_geometry_prior(
    appearance_scores: np.ndarray,
    diagnostics: GeometryDiagnostics,
    config: GeometryConfig,
) -> np.ndarray:
    if not config.enabled or not diagnostics.available or config.score_weight <= 0:
        return appearance_scores.copy()
    weight = min(max(config.score_weight, 0.0), 1.0)
    return (1.0 - weight) * appearance_scores + weight * diagnostics.prior[None, :]
