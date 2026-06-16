from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from config import (
    MIXVPR_CHECKPOINT,
    VPR_BATCH_SIZE,
    VPR_BACKEND_DIR,
    VPR_CACHE_DIR,
    VPR_FILTER_ENABLED,
    VPR_INITIAL_SEARCH_SIZE,
    VPR_MAX_WINDOW_DISTANCE_M,
    VPR_WINDOW_FORWARD,
)


BACKEND_SRC = Path(VPR_BACKEND_DIR).expanduser().resolve()
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

try:
    from dev_matching.config import FilteringConfig, GeometryConfig
    from dev_matching.data import ImageCollection
    from dev_matching.features import DescriptorCache
    from dev_matching.filtering import filter_rosa, identity_filter
    from dev_matching.matchers import build_matcher
    from dev_matching.pipeline.fusion import fuse_matrices
    from dev_matching.pipeline.geometry import analyze_geometry, apply_geometry_prior
    from dev_matching.trajectory import decode_trajectory, local_search_windows
    from dev_matching.utils.device import resolve_device
except ImportError as exc:
    raise RuntimeError(
        f"Cannot import the embedded VPR backend from {BACKEND_SRC}. "
        "The Robot_Visual_Navigation_V3/vpr_backend folder is missing or incomplete."
    ) from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_MODES = ("MixVPR", "Lightweight VPR", "Hybrid VPR")


class VPRMatcher:
    """Frontend adapter for MixVPR, ROSA filtering and local trajectory search."""

    def __init__(
        self,
        database_dir: str | Path,
        metadata_file: str | Path,
        mode: str = "MixVPR",
        checkpoint: str | Path = MIXVPR_CHECKPOINT,
        cache_dir: str | Path = VPR_CACHE_DIR,
        batch_size: int = VPR_BATCH_SIZE,
        filter_enabled: bool = VPR_FILTER_ENABLED,
        device: str = "auto",
    ):
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unknown VPR mode: {mode}")
        self.database_dir = Path(database_dir).expanduser().resolve()
        self.metadata_file = Path(metadata_file).expanduser().resolve()
        self.mode = mode
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.batch_size = int(batch_size)
        self.device = resolve_device(device)
        self.cache = DescriptorCache(cache_dir)
        self.geometry_config = GeometryConfig(enabled=True, score_weight=0.05)
        self.filter_config = FilteringConfig(enabled=filter_enabled)

        self.database = ImageCollection.from_directory(
            "streetview",
            self.database_dir,
            self.metadata_file,
        )
        self.matchers = {
            "spatial_pyramid": build_matcher(
                "spatial_pyramid",
                device=self.device,
                batch_size=self.batch_size,
                image_size=256,
                levels=[1, 2, 4],
            ),
            "color_histogram": build_matcher(
                "color_histogram",
                device=self.device,
                batch_size=self.batch_size,
                bins=32,
            ),
        }
        if mode in {"MixVPR", "Hybrid VPR"}:
            self.matchers["mixvpr"] = build_matcher(
                "mixvpr",
                device=self.device,
                batch_size=self.batch_size,
                checkpoint=self.checkpoint,
            )

        self.database_descriptors = {
            name: self.cache.get_or_compute(
                self.database,
                matcher.name,
                matcher.cache_signature,
                lambda matcher=matcher: matcher.extract([record.path for record in self.database.records]),
            )
            for name, matcher in self.matchers.items()
        }
        self.full_geometry = analyze_geometry(self.database, self.geometry_config)
        if filter_enabled:
            try:
                self.filter_result = filter_rosa(
                    self.database,
                    self.database_descriptors["spatial_pyramid"],
                    self.full_geometry,
                    self.filter_config,
                )
            except ValueError:
                self.filter_result = identity_filter(self.database, method="rosa_fallback")
        else:
            self.filter_result = identity_filter(self.database)
        self.selected_indices = self.filter_result.selected_indices
        self.active_database = self.database.subset(self.selected_indices)
        self.active_geometry = analyze_geometry(self.active_database, self.geometry_config)
        self.route_positions = (
            self.full_geometry.route_positions_m[self.selected_indices]
            if self.full_geometry.route_positions_m is not None
            else self.active_geometry.route_positions_m
        )

    @property
    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "device": self.device,
            "references_before_filtering": len(self.database),
            "references_after_filtering": len(self.active_database),
            "filter_method": self.filter_result.method,
            "filter_counts": self.filter_result.counts_by_reason,
        }

    def _extract_queries(self, collection: ImageCollection) -> dict[str, np.ndarray]:
        paths = [record.path for record in collection.records]
        return {
            name: self.cache.get_or_compute(
                collection,
                matcher.name,
                matcher.cache_signature,
                lambda matcher=matcher: matcher.extract(paths),
            )
            for name, matcher in self.matchers.items()
        }

    def _score_matrices(self, queries: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        matrices = {
            name: matcher.similarity_matrix(
                queries[name],
                self.database_descriptors[name][self.selected_indices],
            )
            for name, matcher in self.matchers.items()
        }
        lightweight = fuse_matrices(
            {
                "color_histogram": matrices["color_histogram"],
                "spatial_pyramid": matrices["spatial_pyramid"],
            },
            {"color_histogram": 0.25, "spatial_pyramid": 0.75},
            mode="rank",
        )
        components = {"lightweight": lightweight}
        if "mixvpr" in matrices:
            components["mixvpr"] = matrices["mixvpr"]

        if self.mode == "MixVPR":
            appearance = matrices["mixvpr"]
        elif self.mode == "Lightweight VPR":
            appearance = lightweight
        else:
            appearance = fuse_matrices(
                {"mixvpr": matrices["mixvpr"], "lightweight": lightweight},
                {"mixvpr": 0.8, "lightweight": 0.2},
                mode="rank",
            )
        final_scores = apply_geometry_prior(appearance, self.active_geometry, self.geometry_config)
        return final_scores, components

    @staticmethod
    def _percentage(value: float) -> float:
        return 100.0 * float(np.clip(value, 0.0, 1.0))

    def _result(self, active_index: int, score: float, components: dict[str, np.ndarray], row: int) -> dict:
        original_index = int(self.selected_indices[active_index])
        record = self.database.records[original_index]
        result = {
            "filename": record.path.name,
            "score": self._percentage(score),
            "lat": record.latitude,
            "lon": record.longitude,
            "match_index": original_index,
            "active_match_index": active_index,
            "mixvpr_score": None,
            "lightweight_score": self._percentage(components["lightweight"][row, active_index]),
            "filter_method": self.filter_result.method,
            "references_before_filtering": len(self.database),
            "references_after_filtering": len(self.active_database),
        }
        if "mixvpr" in components:
            result["mixvpr_score"] = self._percentage(components["mixvpr"][row, active_index])
        return result

    def search(
        self,
        query_image_path: str | Path,
        top_k: int = 5,
        progress_callback: Callable | None = None,
    ) -> list[dict]:
        query_path = Path(query_image_path).expanduser().resolve()
        if not query_path.is_file() or query_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise FileNotFoundError(f"Query image not found: {query_path}")
        query_descriptors = {
            name: matcher.extract([query_path])
            for name, matcher in self.matchers.items()
        }
        scores, components = self._score_matrices(query_descriptors)
        order = np.argsort(scores[0])[::-1][:max(int(top_k), 1)]
        results = [self._result(int(index), float(scores[0, index]), components, 0) for index in order]
        if progress_callback is not None:
            for current, result in enumerate(results, start=1):
                progress_callback(current=current, total=len(results), result=result)
        return results

    def localize_sequence(
        self,
        frames_dir: str | Path,
        top_k: int = 5,
        min_score: float = 0.0,
        output_dir: str | Path | None = None,
        progress_callback: Callable | None = None,
    ) -> list[dict]:
        frames = ImageCollection.from_directory("robot_frames", frames_dir)
        started = time.perf_counter()
        queries = self._extract_queries(frames)
        scores, components = self._score_matrices(queries)
        trajectory = decode_trajectory(
            scores,
            algorithm="local_window_search",
            initial_search_size=VPR_INITIAL_SEARCH_SIZE,
            initial_evidence_frames=3,
            initial_index_penalty=0.04,
            window_backward=0,
            window_forward=VPR_WINDOW_FORWARD,
            max_window_distance_m=VPR_MAX_WINDOW_DISTANCE_M,
            temporal_smoothing=0.65,
            local_stay_penalty=0.01,
            reference_positions=self.route_positions,
        )
        window_starts, window_stops = local_search_windows(
            trajectory,
            len(self.active_database),
            initial_search_size=VPR_INITIAL_SEARCH_SIZE,
            window_backward=0,
            window_forward=VPR_WINDOW_FORWARD,
            reference_positions=self.route_positions,
            max_window_distance_m=VPR_MAX_WINDOW_DISTANCE_M,
        )

        results = []
        total = len(frames)
        for row, frame_record in enumerate(frames.records):
            active_index = int(trajectory[row])
            result = self._result(active_index, float(scores[row, active_index]), components, row)
            result.update(
                {
                    "frame": frame_record.path.name,
                    "best_match": result["filename"],
                    "confidence": "low" if result["score"] < min_score else "accepted",
                    "search_start": int(window_starts[row]),
                    "search_stop": int(window_stops[row]),
                    "search_candidates": int(window_stops[row] - window_starts[row] + 1),
                    "top_candidates": json.dumps(
                        [
                            self.active_database.records[int(index)].path.name
                            for index in np.argsort(scores[row])[::-1][:max(int(top_k), 1)]
                        ]
                    ),
                }
            )
            results.append(result)
            if progress_callback is not None:
                elapsed = time.perf_counter() - started
                remaining = elapsed / max(row + 1, 1) * (total - row - 1)
                progress_callback(
                    current=row + 1,
                    total=total,
                    remaining=remaining,
                    filename=frame_record.path.name,
                )

        if output_dir is not None:
            self._write_outputs(Path(output_dir), frames, queries, scores, trajectory, results)
        return results

    def _write_outputs(
        self,
        output_dir: Path,
        frames: ImageCollection,
        queries: dict[str, np.ndarray],
        scores: np.ndarray,
        trajectory: np.ndarray,
        results: list[dict],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "similarity.npy", scores)
        np.save(output_dir / "trajectory.npy", trajectory)
        for name, descriptors in queries.items():
            np.save(output_dir / f"descriptors_robot_{name}.npy", descriptors)
            np.save(
                output_dir / f"descriptors_streetview_{name}.npy",
                self.database_descriptors[name][self.selected_indices],
            )
        with (output_dir / "localization.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(results[0]) if results else ["frame", "best_match", "score"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        with (output_dir / "filter_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["original_index", "filename", "selected", "reason", "representative_index"])
            for decision, record in zip(self.filter_result.decisions, self.database.records):
                writer.writerow(
                    [
                        decision.original_index,
                        record.path.name,
                        decision.selected,
                        decision.reason,
                        decision.representative_index,
                    ]
                )
        (output_dir / "metadata.json").write_text(
            json.dumps(self.summary, indent=2),
            encoding="utf-8",
        )
