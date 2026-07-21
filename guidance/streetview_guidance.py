"""Build steering targets from MixVPR localization and Street View metadata."""

import csv
import os
from pathlib import Path

import cv2
import numpy as np

from .path_follower import (
    GuidanceConfig,
    StreetViewReferenceTarget,
    VisualPathFollower,
)


def _optional_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _angle_delta_deg(source, target):
    """Signed bearing delta: positive is clockwise/right."""
    return (target - source + 180.0) % 360.0 - 180.0


class StreetViewGuidanceBuilder:
    """Convert sequence-localization rows into DB-guided steering targets."""

    def __init__(
        self,
        database_dir,
        metadata_file,
        guidance_config=None,
        route_lookahead_images=3,
    ):
        self.database_dir = Path(database_dir)
        self.metadata_file = Path(metadata_file)
        self.guidance_config = guidance_config or GuidanceConfig()
        self.route_lookahead_images = max(1, int(route_lookahead_images))
        self.rows = self._load_metadata()
        self.by_filename = {
            os.path.basename(row.get("image_file", "")): row
            for row in self.rows
        }
        self.by_index = {
            int(row["index"]): row
            for row in self.rows
            if str(row.get("index", "")).strip().isdigit()
        }
        self._geometry_cache = {}

    def _load_metadata(self):
        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Street View metadata not found: {self.metadata_file}"
            )
        with self.metadata_file.open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _reference_geometry(self, filename):
        if filename in self._geometry_cache:
            return self._geometry_cache[filename]

        image_path = self.database_dir / filename
        frame = cv2.imread(str(image_path))
        if frame is None:
            geometry = (0.0, 0.0, 0.0)
        else:
            config = GuidanceConfig(
                **{
                    **self.guidance_config.__dict__,
                    "command_smoothing": 1.0,
                    "boundary_smoothing": 1.0,
                    "min_confidence": 0.0,
                    "require_streetview_reference": False,
                }
            )
            result = VisualPathFollower(config).process(frame)
            geometry = (
                result.lateral_error,
                result.heading_error,
                result.confidence,
            )
        self._geometry_cache[filename] = geometry
        return geometry

    def _future_heading(self, match_index, current_heading):
        future_index = min(
            max(self.by_index) if self.by_index else match_index,
            match_index + self.route_lookahead_images,
        )
        future_row = self.by_index.get(future_index)
        if future_row is None:
            return current_heading, 0.0
        future_heading = _optional_float(future_row.get("heading"))
        if current_heading is None or future_heading is None:
            return future_heading, 0.0

        delta = _angle_delta_deg(current_heading, future_heading)
        # A near U-turn at the final duplicated GPS point is treated as route
        # termination, not as an immediate steering request.
        if abs(delta) > 120.0:
            return future_heading, 0.0
        return future_heading, float(np.clip(delta / 90.0, -1.0, 1.0))

    def build_target(self, localization_result):
        filename = (
            localization_result.get("best_match")
            or localization_result.get("filename")
            or ""
        )
        metadata = self.by_filename.get(os.path.basename(filename))
        if metadata is None:
            return None

        match_index = int(
            localization_result.get("match_index", metadata.get("index", 0))
        )
        heading = _optional_float(metadata.get("heading"))
        future_heading, route_turn_error = self._future_heading(
            match_index,
            heading,
        )
        expected_lateral, expected_heading, geometry_confidence = (
            self._reference_geometry(os.path.basename(filename))
        )
        vpr_score = float(localization_result.get("score") or 0.0)
        vpr_confidence = float(np.clip(vpr_score / 40.0, 0.20, 1.0))
        reference_confidence = float(
            np.clip(0.70 * vpr_confidence + 0.30 * geometry_confidence, 0.0, 1.0)
        )

        return StreetViewReferenceTarget(
            filename=os.path.basename(filename),
            match_index=match_index,
            latitude=_optional_float(
                metadata.get("target_lat") or localization_result.get("lat")
            ),
            longitude=_optional_float(
                metadata.get("target_lon") or localization_result.get("lon")
            ),
            heading_deg=heading,
            future_heading_deg=future_heading,
            vpr_score=vpr_score,
            expected_lateral_error=expected_lateral,
            expected_heading_error=expected_heading,
            route_turn_error=route_turn_error,
            confidence=reference_confidence,
        )

    def build_targets(self, localization_results):
        return [self.build_target(result) for result in localization_results]
