from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dev_matching.config import AppConfig, DatasetPair
from dev_matching.data import ImageCollection
from dev_matching.features import DescriptorCache
from dev_matching.filtering import filter_rosa, identity_filter
from dev_matching.matchers import build_matcher
from dev_matching.trajectory import decode_trajectory, local_search_windows
from dev_matching.utils.device import resolve_device
from dev_matching.visualization import write_filter_overview, write_match_sheets

from .artifacts import (
    write_image_metadata_csv,
    write_localization_csv,
    write_rankings_csv,
    write_similarity_csv,
    write_trajectory_csv,
    write_geometry_csv,
    write_filter_manifest_csv,
    write_search_windows_csv,
)
from .fusion import fuse_matrices
from .geometry import analyze_geometry, apply_geometry_prior

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentResult:
    pair: DatasetPair
    scores: np.ndarray
    method_scores: dict[str, np.ndarray]
    trajectory: np.ndarray
    output_dir: Path


class VPRExperiment:
    def __init__(self, config: AppConfig):
        self.config = config
        self.cache = DescriptorCache(config.cache_dir)
        self.device = resolve_device(config.device)

    def run_pair(self, pair: DatasetPair) -> ExperimentResult:
        started = time.perf_counter()
        robot = ImageCollection.from_directory(pair.robot_name, pair.robot_dir, pair.robot_metadata_csv)
        full_streetview = ImageCollection.from_directory(
            pair.streetview_name, pair.streetview_dir, pair.streetview_metadata_csv
        )
        enabled = [method for method in self.config.methods if method.enabled and method.weight > 0]
        if not enabled:
            raise ValueError("At least one positively weighted method must be enabled")
        matchers = {}
        query_descriptors: dict[str, np.ndarray] = {}
        database_descriptors: dict[str, np.ndarray] = {}
        for method in enabled:
            LOGGER.info(
                "Extracting/caching %s descriptors for %s x %s",
                method.name,
                robot.name,
                full_streetview.name,
            )
            matcher = build_matcher(
                method.name,
                device=self.device,
                batch_size=self.config.batch_size,
                **method.params,
            )
            matchers[method.name] = matcher
            query_descriptors[method.name] = self.cache.get_or_compute(
                robot, matcher.name, matcher.cache_signature,
                lambda m=matcher: m.extract([r.path for r in robot.records]),
            )
            database_descriptors[method.name] = self.cache.get_or_compute(
                full_streetview, matcher.name, matcher.cache_signature,
                lambda m=matcher: m.extract([r.path for r in full_streetview.records]),
            )

        full_geometry = analyze_geometry(full_streetview, self.config.geometry)
        fc = self.config.filtering
        if fc.enabled:
            if fc.descriptor_method not in database_descriptors:
                method_spec = next((method for method in self.config.methods if method.name == fc.descriptor_method), None)
                params = method_spec.params if method_spec is not None else {}
                filter_matcher = build_matcher(
                    fc.descriptor_method,
                    device=self.device,
                    batch_size=self.config.batch_size,
                    **params,
                )
                database_descriptors[fc.descriptor_method] = self.cache.get_or_compute(
                    full_streetview,
                    filter_matcher.name,
                    filter_matcher.cache_signature,
                    lambda m=filter_matcher: m.extract([record.path for record in full_streetview.records]),
                )
            filter_result = filter_rosa(
                full_streetview,
                database_descriptors[fc.descriptor_method],
                full_geometry,
                fc,
            )
        else:
            filter_result = identity_filter(full_streetview)
        selected_indices = filter_result.selected_indices
        streetview = full_streetview.subset(selected_indices)
        LOGGER.info(
            "ROSA filtering retained %d/%d references: %s",
            len(streetview),
            len(full_streetview),
            filter_result.counts_by_reason,
        )

        matrices: dict[str, np.ndarray] = {}
        full_matrices: dict[str, np.ndarray] = {}
        for method in enabled:
            matcher = matchers[method.name]
            query = query_descriptors[method.name]
            database = database_descriptors[method.name]
            matrices[method.name] = matcher.similarity_matrix(query, database[selected_indices])
            if fc.preserve_unfiltered_matrix:
                full_matrices[method.name] = matcher.similarity_matrix(query, database)
        appearance_scores = fuse_matrices(
            matrices,
            {method.name: method.weight for method in enabled},
            mode=self.config.fusion,
        )
        geometry = analyze_geometry(streetview, self.config.geometry)
        scores = apply_geometry_prior(appearance_scores, geometry, self.config.geometry)
        active_route_positions = (
            full_geometry.route_positions_m[selected_indices]
            if full_geometry.route_positions_m is not None
            else geometry.route_positions_m
        )
        tc = self.config.trajectory
        trajectory = decode_trajectory(
            scores,
            algorithm=tc.algorithm,
            max_step=tc.max_step,
            stay_penalty=tc.stay_penalty,
            jump_penalty=tc.jump_penalty,
            backward_penalty=tc.backward_penalty,
            emission_temperature=tc.emission_temperature,
            reference_positions=active_route_positions,
            progress_weight=tc.progress_weight,
            progress_sigma_fraction=tc.progress_sigma_fraction,
            speed_penalty=tc.speed_penalty,
            initial_search_size=tc.initial_search_size,
            initial_evidence_frames=tc.initial_evidence_frames,
            initial_index_penalty=tc.initial_index_penalty,
            window_backward=tc.window_backward,
            window_forward=tc.window_forward,
            max_window_distance_m=tc.max_window_distance_m,
            temporal_smoothing=tc.temporal_smoothing,
            local_stay_penalty=tc.local_stay_penalty,
            allowed_states=(
                geometry.primary_route_mask
                if self.config.geometry.restrict_to_primary_heading_segment
                else None
            ),
        )
        if tc.algorithm == "local_window_search":
            window_starts, window_stops = local_search_windows(
                trajectory,
                len(streetview),
                initial_search_size=tc.initial_search_size,
                window_backward=tc.window_backward,
                window_forward=tc.window_forward,
                reference_positions=active_route_positions,
                max_window_distance_m=tc.max_window_distance_m,
            )
        else:
            window_starts = np.zeros(len(robot), dtype=np.int64)
            window_stops = np.full(len(robot), len(streetview) - 1, dtype=np.int64)
        output = self.config.output_dir / f"{robot.name}__{streetview.name}"
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "similarity.npy", scores)
        np.save(output / "appearance_similarity.npy", appearance_scores)
        np.save(output / "geometry_prior.npy", geometry.prior)
        descriptor_dimensions = {}
        for name in matchers:
            query = query_descriptors[name]
            database = database_descriptors[name]
            descriptor_dimensions[name] = int(query.shape[1])
            np.save(output / f"descriptors_robot_{name}.npy", query)
            np.save(output / f"descriptors_streetview_{name}.npy", database[selected_indices])
            np.save(output / f"descriptors_streetview_unfiltered_{name}.npy", database)
        for name, matrix in matrices.items():
            np.save(output / f"similarity_{name}.npy", matrix)
        if full_matrices:
            full_appearance_scores = fuse_matrices(
                full_matrices,
                {method.name: method.weight for method in enabled},
                mode=self.config.fusion,
            )
            full_scores = apply_geometry_prior(full_appearance_scores, full_geometry, self.config.geometry)
            np.save(output / "similarity_unfiltered.npy", full_scores)
            for name, matrix in full_matrices.items():
                np.save(output / f"similarity_unfiltered_{name}.npy", matrix)
            write_similarity_csv(
                output / "similarity_unfiltered.csv",
                robot,
                full_streetview,
                full_scores,
                appearance_scores=full_appearance_scores,
                geometry_prior=full_geometry.prior,
            )
        write_similarity_csv(
            output / "similarity.csv",
            robot,
            streetview,
            scores,
            appearance_scores=appearance_scores,
            geometry_prior=geometry.prior,
        )
        write_rankings_csv(output / "rankings_top10.csv", robot, streetview, scores)
        write_trajectory_csv(output / "trajectory.csv", robot, streetview, trajectory, scores)
        write_search_windows_csv(
            output / "search_windows.csv",
            robot,
            streetview,
            scores,
            trajectory,
            window_starts,
            window_stops,
        )
        write_image_metadata_csv(output / "image_metadata.csv", robot, full_streetview)
        write_geometry_csv(output / "geometry_diagnostics.csv", full_streetview, full_geometry)
        write_filter_manifest_csv(output / "filter_manifest.csv", full_streetview, filter_result)
        write_filter_overview(output / "filter_overview.png", full_streetview, filter_result)
        write_localization_csv(
            output / "localization.csv",
            robot,
            streetview,
            scores,
            trajectory,
            route_positions_m=active_route_positions,
            window_starts=window_starts,
            window_stops=window_stops,
        )
        write_match_sheets(
            output / "matches",
            robot,
            streetview,
            scores,
            trajectory,
            window_starts=window_starts,
            window_stops=window_stops,
        )
        metadata = {
            "robot_database": robot.name,
            "streetview_database": streetview.name,
            "shape": list(scores.shape),
            "unfiltered_shape": [len(robot), len(full_streetview)],
            "filtering_method": filter_result.method,
            "streetview_images_before_filtering": len(full_streetview),
            "streetview_images_after_filtering": len(streetview),
            "streetview_images_rejected": len(full_streetview) - len(streetview),
            "streetview_reduction_ratio": 1.0 - len(streetview) / len(full_streetview),
            "filtering_counts": filter_result.counts_by_reason,
            "trajectory_algorithm": tc.algorithm,
            "initial_search_size": tc.initial_search_size,
            "initial_evidence_frames": tc.initial_evidence_frames,
            "local_window_backward": tc.window_backward,
            "local_window_forward": tc.window_forward,
            "max_window_distance_m": tc.max_window_distance_m,
            "methods": {m.name: m.weight for m in enabled},
            "descriptor_dimensions": descriptor_dimensions,
            "fusion": self.config.fusion,
            "geometry_prior_enabled": self.config.geometry.enabled and geometry.available,
            "geometry_score_weight": self.config.geometry.score_weight,
            "device": self.device,
            "robot_images_with_gps": sum(record.has_position for record in robot.records),
            "streetview_images_with_gps": sum(record.has_position for record in full_streetview.records),
            "streetview_images_with_route_csv": sum(record.route_index is not None for record in full_streetview.records),
            "streetview_unique_panoramas": len({record.pano_id for record in full_streetview.records if record.pano_id}),
            "streetview_metadata_csv": str(pair.streetview_metadata_csv) if pair.streetview_metadata_csv else None,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if metadata["streetview_images_with_gps"] == 0:
            LOGGER.warning(
                "No Street View GPS metadata found for %s; metric distances will be empty",
                streetview.name,
            )
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        LOGGER.info("Completed %s x %s in %.2fs", robot.name, streetview.name, metadata["elapsed_seconds"])
        return ExperimentResult(pair, scores, matrices, trajectory, output)

    def run_all(self) -> list[ExperimentResult]:
        return [self.run_pair(pair) for pair in self.config.pairs]
