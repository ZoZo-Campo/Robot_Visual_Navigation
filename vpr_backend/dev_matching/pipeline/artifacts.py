from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from dev_matching.data import ImageCollection
from dev_matching.evaluation import haversine_meters
from dev_matching.filtering import RosaFilterResult
from .geometry import GeometryDiagnostics


def write_similarity_csv(
    path: str | Path,
    robot: ImageCollection,
    streetview: ImageCollection,
    scores: np.ndarray,
    appearance_scores: np.ndarray | None = None,
    geometry_prior: np.ndarray | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if scores.shape != (len(robot), len(streetview)):
        raise ValueError("Score matrix shape does not match collections")
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "robot_database", "streetview_database", "frame", "streetview", "score",
            "appearance_score", "geometry_prior",
        ])
        for i, frame in enumerate(robot.records):
            for j, reference in enumerate(streetview.records):
                writer.writerow([
                    robot.name,
                    streetview.name,
                    frame.id,
                    reference.id,
                    float(scores[i, j]),
                    float(appearance_scores[i, j]) if appearance_scores is not None else float(scores[i, j]),
                    float(geometry_prior[j]) if geometry_prior is not None else 1.0,
                ])


def write_trajectory_csv(path: str | Path, robot: ImageCollection, streetview: ImageCollection, states: np.ndarray, scores: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "streetview", "streetview_index", "score"])
        for i, state in enumerate(states):
            writer.writerow([robot.records[i].id, streetview.records[int(state)].id, int(state), float(scores[i, state])])


def write_search_windows_csv(
    path: str | Path,
    robot: ImageCollection,
    streetview: ImageCollection,
    scores: np.ndarray,
    trajectory: np.ndarray,
    window_starts: np.ndarray,
    window_stops: np.ndarray,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "frame", "window_start_index", "window_start_streetview", "window_stop_index",
            "window_stop_streetview", "candidate_count", "selected_index", "selected_streetview",
            "selected_score", "selected_local_rank",
        ])
        for frame, state in enumerate(trajectory):
            start, stop = int(window_starts[frame]), int(window_stops[frame])
            candidates = np.arange(start, stop + 1)
            selected_rank = int(np.sum(scores[frame, candidates] > scores[frame, state]) + 1)
            writer.writerow([
                robot.records[frame].id,
                start,
                streetview.records[start].id,
                stop,
                streetview.records[stop].id,
                len(candidates),
                int(state),
                streetview.records[int(state)].id,
                float(scores[frame, state]),
                selected_rank,
            ])


def write_rankings_csv(
    path: str | Path,
    robot: ImageCollection,
    streetview: ImageCollection,
    scores: np.ndarray,
    top_k: int = 10,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-scores, axis=1)[:, : min(top_k, len(streetview))]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "rank", "streetview", "streetview_index", "score"])
        for query_index, ranked in enumerate(order):
            for rank, reference_index in enumerate(ranked, start=1):
                writer.writerow([
                    robot.records[query_index].id,
                    rank,
                    streetview.records[int(reference_index)].id,
                    int(reference_index),
                    float(scores[query_index, reference_index]),
                ])


def write_image_metadata_csv(path: str | Path, *collections: ImageCollection) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "database", "id", "filename", "route_index", "target_latitude", "target_longitude",
            "pano_id", "pano_latitude", "pano_longitude", "distance_to_pano_m", "heading", "timestamp",
        ])
        for collection in collections:
            for record in collection.records:
                writer.writerow([
                    collection.name,
                    record.id,
                    record.path.name,
                    record.route_index if record.route_index is not None else "",
                    record.target_latitude if record.target_latitude is not None else (
                        record.latitude if record.latitude is not None else ""
                    ),
                    record.target_longitude if record.target_longitude is not None else (
                        record.longitude if record.longitude is not None else ""
                    ),
                    record.pano_id or "",
                    record.pano_latitude if record.pano_latitude is not None else "",
                    record.pano_longitude if record.pano_longitude is not None else "",
                    record.distance_to_pano_m if record.distance_to_pano_m is not None else "",
                    record.heading if record.heading is not None else "",
                    record.timestamp or "",
                ])


def write_geometry_csv(
    path: str | Path,
    streetview: ImageCollection,
    diagnostics: GeometryDiagnostics,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "streetview", "route_index", "route_position_m", "route_bearing_deg", "image_heading_deg",
            "heading_error_deg", "heading_alignment", "distance_to_pano_m", "pano_distance_quality",
            "geometry_prior", "route_segment", "primary_route_segment", "pano_id",
        ])
        for index, record in enumerate(streetview.records):
            writer.writerow([
                record.id,
                record.route_index if record.route_index is not None else index,
                float(diagnostics.route_positions_m[index]) if diagnostics.route_positions_m is not None else "",
                float(diagnostics.route_bearings_deg[index]) if not np.isnan(diagnostics.route_bearings_deg[index]) else "",
                record.heading if record.heading is not None else "",
                float(diagnostics.heading_errors_deg[index]) if not np.isnan(diagnostics.heading_errors_deg[index]) else "",
                float(diagnostics.heading_alignment[index]),
                record.distance_to_pano_m if record.distance_to_pano_m is not None else "",
                float(diagnostics.pano_distance_quality[index]),
                float(diagnostics.prior[index]),
                int(diagnostics.route_segments[index]),
                bool(diagnostics.primary_route_mask[index]),
                record.pano_id or "",
            ])


def write_filter_manifest_csv(
    path: str | Path,
    streetview: ImageCollection,
    result: RosaFilterResult,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    selected_lookup = {int(original): filtered for filtered, original in enumerate(result.selected_indices)}
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "original_index", "filtered_index", "streetview", "filename", "selected", "reason",
            "representative_original_index", "representative_streetview", "route_index",
            "route_position_m", "distance_from_last_kept_m", "similarity_to_last_kept",
            "pano_id", "heading", "distance_to_pano_m",
        ])
        for decision in result.decisions:
            record = streetview.records[decision.original_index]
            representative = (
                streetview.records[decision.representative_index]
                if decision.representative_index is not None
                else None
            )
            writer.writerow([
                decision.original_index,
                selected_lookup.get(decision.original_index, ""),
                record.id,
                record.path.name,
                decision.selected,
                decision.reason,
                decision.representative_index if decision.representative_index is not None else "",
                representative.id if representative is not None else "",
                record.route_index if record.route_index is not None else "",
                decision.route_position_m if decision.route_position_m is not None else "",
                decision.distance_from_last_kept_m if decision.distance_from_last_kept_m is not None else "",
                decision.similarity_to_last_kept if decision.similarity_to_last_kept is not None else "",
                record.pano_id or "",
                record.heading if record.heading is not None else "",
                record.distance_to_pano_m if record.distance_to_pano_m is not None else "",
            ])


def write_localization_csv(
    path: str | Path,
    robot: ImageCollection,
    streetview: ImageCollection,
    scores: np.ndarray,
    trajectory: np.ndarray,
    route_positions_m: np.ndarray | None = None,
    window_starts: np.ndarray | None = None,
    window_stops: np.ndarray | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    previous_reference = None
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "frame", "matched_streetview", "matched_filename", "score",
            "top1_margin",
            "estimated_route_position_m", "estimated_target_latitude", "estimated_target_longitude",
            "pano_id", "pano_latitude", "pano_longitude", "distance_target_to_pano_m", "heading",
            "robot_latitude", "robot_longitude", "error_meters", "distance_from_previous_match_meters",
            "selected_visual_rank", "selected_score_gap_to_top1", "selected_local_rank",
            "local_score_gap_to_top1", "window_start_index", "window_stop_index",
            "candidate_spread_m", "estimated_uncertainty_m", "sequence_confidence",
        ])
        for query_index, state in enumerate(trajectory):
            query = robot.records[query_index]
            reference = streetview.records[int(state)]
            sorted_scores = np.sort(scores[query_index])
            margin = float(sorted_scores[-1] - sorted_scores[-2]) if len(sorted_scores) > 1 else 1.0
            plausible = np.flatnonzero(scores[query_index] >= float(sorted_scores[-1]) - 0.05)
            selected_rank = int(np.sum(scores[query_index] > scores[query_index, state]) + 1)
            selected_gap = float(scores[query_index, state] - sorted_scores[-1])
            window_start = int(window_starts[query_index]) if window_starts is not None else 0
            window_stop = int(window_stops[query_index]) if window_stops is not None else len(streetview) - 1
            local_scores_for_rank = scores[query_index, window_start:window_stop + 1]
            local_top1 = float(np.max(local_scores_for_rank))
            selected_local_rank = int(np.sum(local_scores_for_rank > scores[query_index, state]) + 1)
            local_gap = float(scores[query_index, state] - local_top1)
            candidate_spread = ""
            sequence_confidence = ""
            uncertainty = ""
            if route_positions_m is not None:
                selected_position = route_positions_m[int(state)]
                local_mask = np.abs(route_positions_m - selected_position) <= 25.0
                local_plausible = plausible[local_mask[plausible]]
                if len(local_plausible):
                    candidate_spread = float(
                        np.max(np.abs(route_positions_m[local_plausible] - selected_position))
                    )
                local_indices = np.flatnonzero(local_mask)
                local_scores = scores[query_index, local_indices]
                weights = np.exp((local_scores - np.max(local_scores)) / 0.05)
                weights /= np.maximum(weights.sum(), 1e-12)
                route_std = float(
                    np.sqrt(np.sum(weights * (route_positions_m[local_indices] - selected_position) ** 2))
                )
                uncertainty = route_std + float(reference.distance_to_pano_m or 0.0)
                sequence_confidence = float(np.exp(-uncertainty / 20.0))
            elif reference.distance_to_pano_m is not None:
                uncertainty = float(reference.distance_to_pano_m)
            error = ""
            if query.has_position and reference.has_position:
                error = haversine_meters(query.latitude, query.longitude, reference.latitude, reference.longitude)
            step_distance = ""
            if previous_reference is not None and previous_reference.has_position and reference.has_position:
                step_distance = haversine_meters(
                    previous_reference.latitude,
                    previous_reference.longitude,
                    reference.latitude,
                    reference.longitude,
                )
            writer.writerow([
                query.id, reference.id, reference.path.name, float(scores[query_index, state]), margin,
                float(route_positions_m[int(state)]) if route_positions_m is not None else "",
                reference.latitude if reference.latitude is not None else "",
                reference.longitude if reference.longitude is not None else "",
                reference.pano_id or "",
                reference.pano_latitude if reference.pano_latitude is not None else "",
                reference.pano_longitude if reference.pano_longitude is not None else "",
                reference.distance_to_pano_m if reference.distance_to_pano_m is not None else "",
                reference.heading if reference.heading is not None else "",
                query.latitude if query.latitude is not None else "",
                query.longitude if query.longitude is not None else "",
                error, step_distance, selected_rank, selected_gap, selected_local_rank,
                local_gap, window_start, window_stop,
                candidate_spread, uncertainty, sequence_confidence,
            ])
            previous_reference = reference
