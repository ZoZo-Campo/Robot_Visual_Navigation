"""Street View DB-guided path following with a ROS-compatible command output.

The controller compares current path geometry with the route-facing Street
View image selected by MixVPR, adds the upcoming metadata heading change and
uses local path boundaries for short-term stability. The command follows the
ROS convention: positive ``angular_z`` turns left and negative turns right.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class GuidanceConfig:
    roi_top_ratio: float = 0.42
    lookahead_ratio: float = 0.62
    near_ratio: float = 0.88
    lateral_gain: float = 0.75
    heading_gain: float = 0.35
    max_angular_z: float = 0.60
    command_smoothing: float = 0.28
    boundary_smoothing: float = 0.30
    deadband: float = 0.035
    min_confidence: float = 0.18
    nominal_linear_x: float = 0.25
    max_missing_frames: int = 10
    reference_alignment_gain: float = 0.75
    route_turn_gain: float = 0.45
    local_stability_gain: float = 0.20
    require_streetview_reference: bool = False


@dataclass
class StreetViewReferenceTarget:
    filename: str
    match_index: int
    latitude: Optional[float]
    longitude: Optional[float]
    heading_deg: Optional[float]
    future_heading_deg: Optional[float]
    vpr_score: float
    expected_lateral_error: float
    expected_heading_error: float
    route_turn_error: float
    confidence: float


@dataclass
class GuidanceResult:
    frame_index: int
    timestamp_s: float
    camera_center_x_px: float
    path_center_x_px: Optional[float]
    lateral_error: float
    heading_error: float
    angular_z: float
    linear_x: float
    confidence: float
    instruction: str
    left_boundary_detected: bool
    right_boundary_detected: bool
    control_source: str
    reference_filename: str
    reference_match_index: Optional[int]
    reference_latitude: Optional[float]
    reference_longitude: Optional[float]
    reference_heading_deg: Optional[float]
    reference_vpr_score: float
    reference_alignment_error: float
    route_turn_error: float
    reference_confidence: float
    annotated_frame: np.ndarray

    def as_csv_row(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": f"{self.timestamp_s:.3f}",
            "camera_center_x_px": f"{self.camera_center_x_px:.2f}",
            "path_center_x_px": (
                "" if self.path_center_x_px is None
                else f"{self.path_center_x_px:.2f}"
            ),
            "lateral_error": f"{self.lateral_error:.6f}",
            "heading_error": f"{self.heading_error:.6f}",
            "angular_z_rad_s": f"{self.angular_z:.6f}",
            "linear_x_m_s": f"{self.linear_x:.6f}",
            "confidence": f"{self.confidence:.6f}",
            "instruction": self.instruction,
            "left_boundary_detected": int(self.left_boundary_detected),
            "right_boundary_detected": int(self.right_boundary_detected),
            "control_source": self.control_source,
            "reference_filename": self.reference_filename,
            "reference_match_index": (
                "" if self.reference_match_index is None
                else self.reference_match_index
            ),
            "reference_latitude": (
                "" if self.reference_latitude is None
                else f"{self.reference_latitude:.8f}"
            ),
            "reference_longitude": (
                "" if self.reference_longitude is None
                else f"{self.reference_longitude:.8f}"
            ),
            "reference_heading_deg": (
                "" if self.reference_heading_deg is None
                else f"{self.reference_heading_deg:.2f}"
            ),
            "reference_vpr_score": f"{self.reference_vpr_score:.4f}",
            "reference_alignment_error": f"{self.reference_alignment_error:.6f}",
            "route_turn_error": f"{self.route_turn_error:.6f}",
            "reference_confidence": f"{self.reference_confidence:.6f}",
        }


class VisualPathFollower:
    """Stateful path-centre estimator and proportional steering controller."""

    def __init__(self, config: Optional[GuidanceConfig] = None):
        self.config = config or GuidanceConfig()
        self._left_points = None
        self._right_points = None
        self._path_width = None
        self._angular_z = 0.0
        self._missing_frames = 0

    def reset(self):
        self._left_points = None
        self._right_points = None
        self._path_width = None
        self._angular_z = 0.0
        self._missing_frames = 0

    @staticmethod
    def _weighted_median(values, weights):
        order = np.argsort(values)
        sorted_values = np.asarray(values)[order]
        sorted_weights = np.asarray(weights)[order]
        cutoff = sorted_weights.sum() * 0.5
        return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff)])

    def _region_edges(self, frame):
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        median = float(np.median(gray))
        lower = int(max(35, 0.66 * median))
        upper = int(min(220, max(lower + 35, 1.33 * median)))
        edges = cv2.Canny(gray, lower, upper)

        roi_top = int(height * self.config.roi_top_ratio)
        mask = np.zeros_like(edges)
        polygon = np.array(
            [[
                (int(width * 0.08), height - 1),
                (int(width * 0.35), roi_top),
                (int(width * 0.65), roi_top),
                (int(width * 0.92), height - 1),
            ]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def _line_segments(self, edges):
        height, width = edges.shape
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(30, int(width * 0.035)),
            minLineLength=max(35, int(width * 0.045)),
            maxLineGap=max(20, int(width * 0.025)),
        )
        if lines is None:
            return [], []

        y_near = height * self.config.near_ratio
        centre = width * 0.5
        left = []
        right = []

        for item in lines[:, 0]:
            x1, y1, x2, y2 = [float(value) for value in item]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 2 or abs(dy) < 10:
                continue
            slope = dy / dx
            if not 0.35 <= abs(slope) <= 6.0:
                continue

            x_near = x1 + (y_near - y1) * dx / dy
            length = float(np.hypot(dx, dy))
            record = (x1, y1, x2, y2, x_near, length)

            if slope < 0 and x_near < centre * 1.05:
                left.append(record)
            elif slope > 0 and x_near > centre * 0.95:
                right.append(record)

        return left, right

    def _fit_boundary(self, segments, side, height, width):
        if not segments:
            return None, 0.0

        expected = width * (0.20 if side == "left" else 0.80)
        x_near_values = np.array([segment[4] for segment in segments])
        lengths = np.array([segment[5] for segment in segments])
        proximity = np.clip(1.0 - np.abs(x_near_values - expected) / (0.55 * width), 0.1, 1.0)
        anchor = self._weighted_median(x_near_values, lengths * proximity)
        cluster_limit = width * 0.14
        selected = [segment for segment in segments if abs(segment[4] - anchor) <= cluster_limit]
        if not selected:
            selected = segments

        ys = []
        xs = []
        weights = []
        for x1, y1, x2, y2, _, length in selected:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
            weights.extend([length, length])

        if len(xs) < 2:
            return None, 0.0

        coefficient = np.polyfit(
            np.asarray(ys),
            np.asarray(xs),
            1,
            w=np.sqrt(np.asarray(weights)),
        )
        y_far = height * self.config.lookahead_ratio
        y_near = height * self.config.near_ratio
        x_far = float(np.polyval(coefficient, y_far))
        x_near = float(np.polyval(coefficient, y_near))

        margin = width * 0.25
        if not (-margin <= x_far <= width + margin and -margin <= x_near <= width + margin):
            return None, 0.0

        support = min(1.0, sum(segment[5] for segment in selected) / (height * 1.2))
        return np.array([x_far, x_near], dtype=np.float32), support

    @staticmethod
    def _smooth_points(previous, current, alpha):
        if current is None:
            return previous
        if previous is None:
            return current
        return previous * (1.0 - alpha) + current * alpha

    def _resolve_boundaries(self, left, right, width):
        detected_left = left is not None
        detected_right = right is not None

        if left is not None and right is not None:
            measured_width = right - left
            valid = (
                width * 0.12 <= measured_width[0] <= width * 0.85
                and width * 0.30 <= measured_width[1] <= width * 1.45
            )
            if not valid:
                if self._left_points is not None and self._right_points is not None:
                    left_distance = np.mean(np.abs(left - self._left_points))
                    right_distance = np.mean(np.abs(right - self._right_points))
                    if left_distance > right_distance:
                        left = None
                        detected_left = False
                    else:
                        right = None
                        detected_right = False

        if left is not None and right is not None:
            width_now = right - left
            self._path_width = self._smooth_points(
                self._path_width,
                width_now,
                self.config.boundary_smoothing,
            )
        elif left is not None or right is not None:
            # A first frame can contain only one visible curb. Start from a
            # conservative perspective width, then replace it with measured
            # values as soon as both boundaries become visible.
            if self._path_width is None:
                self._path_width = np.array(
                    [width * 0.35, width * 0.65],
                    dtype=np.float32,
                )
            if left is not None:
                right = left + self._path_width
            elif right is not None:
                left = right - self._path_width

        any_detected = detected_left or detected_right
        if any_detected:
            self._missing_frames = 0
        else:
            self._missing_frames += 1
            if self._missing_frames <= self.config.max_missing_frames:
                left = self._left_points
                right = self._right_points
            else:
                left = None
                right = None

        self._left_points = self._smooth_points(
            self._left_points,
            left,
            self.config.boundary_smoothing,
        )
        self._right_points = self._smooth_points(
            self._right_points,
            right,
            self.config.boundary_smoothing,
        )
        return self._left_points, self._right_points, detected_left, detected_right

    def process(
        self,
        frame,
        frame_index=0,
        timestamp_s=0.0,
        reference_target: Optional[StreetViewReferenceTarget] = None,
    ):
        if frame is None or frame.size == 0:
            raise ValueError("The guidance frame is empty.")

        height, width = frame.shape[:2]
        edges = self._region_edges(frame)
        left_segments, right_segments = self._line_segments(edges)
        left, left_support = self._fit_boundary(left_segments, "left", height, width)
        right, right_support = self._fit_boundary(right_segments, "right", height, width)
        left, right, left_detected, right_detected = self._resolve_boundaries(
            left, right, width
        )

        camera_center = width * 0.5
        confidence = 0.0
        path_center = None
        lateral_error = 0.0
        heading_error = 0.0
        raw_angular_z = 0.0
        reference_alignment_error = 0.0
        route_turn_error = 0.0
        reference_confidence = 0.0
        control_source = "LOCAL_VISUAL_FALLBACK"

        if left is not None and right is not None:
            path_far = float((left[0] + right[0]) * 0.5)
            path_near = float((left[1] + right[1]) * 0.5)
            path_center = path_far
            lateral_error = (path_near - camera_center) / (width * 0.5)
            heading_error = (path_far - path_near) / (width * 0.5)

            detection_score = 0.45 * float(left_detected) + 0.45 * float(right_detected)
            support_score = 0.10 * min(1.0, (left_support + right_support) * 0.5)
            confidence = detection_score + support_score
            if not left_detected and not right_detected:
                confidence = max(
                    0.0,
                    0.16 * (1.0 - self._missing_frames / max(1, self.config.max_missing_frames)),
                )

            local_steering_error = (
                self.config.lateral_gain * lateral_error
                + self.config.heading_gain * heading_error
            )

            steering_error = local_steering_error
            if reference_target is not None:
                expected_error = (
                    self.config.lateral_gain
                    * reference_target.expected_lateral_error
                    + self.config.heading_gain
                    * reference_target.expected_heading_error
                )
                reference_alignment_error = local_steering_error - expected_error
                route_turn_error = reference_target.route_turn_error
                reference_confidence = reference_target.confidence
                steering_error = (
                    self.config.reference_alignment_gain
                    * reference_alignment_error
                    * reference_confidence
                    + self.config.route_turn_gain
                    * route_turn_error
                    + self.config.local_stability_gain
                    * local_steering_error
                )
                control_source = "STREETVIEW_DB"

            if abs(steering_error) < self.config.deadband:
                steering_error = 0.0

            # ROS yaw convention: positive is left, negative is right.
            raw_angular_z = -float(
                np.clip(
                    steering_error,
                    -self.config.max_angular_z,
                    self.config.max_angular_z,
                )
            )

        reference_missing = (
            self.config.require_streetview_reference
            and reference_target is None
        )
        if reference_missing:
            target_angular_z = 0.0
            linear_x = 0.0
            instruction = "STOP - NO STREETVIEW MATCH"
            control_source = "NO_DB_MATCH"
        elif confidence < self.config.min_confidence:
            target_angular_z = 0.0
            linear_x = 0.0
            instruction = "STOP - PATH LOST"
        else:
            target_angular_z = raw_angular_z
            turn_ratio = abs(target_angular_z) / max(self.config.max_angular_z, 1e-6)
            linear_x = self.config.nominal_linear_x * max(0.35, 1.0 - 0.65 * turn_ratio)
            if target_angular_z > 0.05:
                instruction = "TURN LEFT"
            elif target_angular_z < -0.05:
                instruction = "TURN RIGHT"
            else:
                instruction = "GO STRAIGHT"

        alpha = self.config.command_smoothing
        self._angular_z = self._angular_z * (1.0 - alpha) + target_angular_z * alpha
        if abs(self._angular_z) < 0.005:
            self._angular_z = 0.0

        annotated = self._draw_overlay(
            frame,
            left,
            right,
            path_center,
            confidence,
            instruction,
            lateral_error,
            heading_error,
            self._angular_z,
            linear_x,
            control_source,
            reference_target,
            reference_alignment_error,
            route_turn_error,
        )

        return GuidanceResult(
            frame_index=int(frame_index),
            timestamp_s=float(timestamp_s),
            camera_center_x_px=camera_center,
            path_center_x_px=path_center,
            lateral_error=float(lateral_error),
            heading_error=float(heading_error),
            angular_z=float(self._angular_z),
            linear_x=float(linear_x),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            instruction=instruction,
            left_boundary_detected=left_detected,
            right_boundary_detected=right_detected,
            control_source=control_source,
            reference_filename=(
                reference_target.filename if reference_target is not None else ""
            ),
            reference_match_index=(
                reference_target.match_index if reference_target is not None else None
            ),
            reference_latitude=(
                reference_target.latitude if reference_target is not None else None
            ),
            reference_longitude=(
                reference_target.longitude if reference_target is not None else None
            ),
            reference_heading_deg=(
                reference_target.heading_deg if reference_target is not None else None
            ),
            reference_vpr_score=(
                reference_target.vpr_score if reference_target is not None else 0.0
            ),
            reference_alignment_error=float(reference_alignment_error),
            route_turn_error=float(route_turn_error),
            reference_confidence=float(reference_confidence),
            annotated_frame=annotated,
        )

    def _draw_overlay(
        self,
        frame,
        left,
        right,
        path_center,
        confidence,
        instruction,
        lateral_error,
        heading_error,
        angular_z,
        linear_x,
        control_source,
        reference_target,
        reference_alignment_error,
        route_turn_error,
    ):
        output = frame.copy()
        height, width = output.shape[:2]
        y_far = int(height * self.config.lookahead_ratio)
        y_near = int(height * self.config.near_ratio)
        camera_x = int(width * 0.5)

        if left is not None and right is not None:
            overlay = output.copy()
            polygon = np.array(
                [[
                    (int(np.clip(left[0], 0, width - 1)), y_far),
                    (int(np.clip(left[1], 0, width - 1)), y_near),
                    (int(np.clip(right[1], 0, width - 1)), y_near),
                    (int(np.clip(right[0], 0, width - 1)), y_far),
                ]],
                dtype=np.int32,
            )
            cv2.fillPoly(overlay, polygon, (30, 170, 70))
            cv2.addWeighted(overlay, 0.22, output, 0.78, 0, output)
            cv2.line(output, (int(left[0]), y_far), (int(left[1]), y_near), (0, 220, 255), 5)
            cv2.line(output, (int(right[0]), y_far), (int(right[1]), y_near), (0, 220, 255), 5)

            path_near = int((left[1] + right[1]) * 0.5)
            path_far = int(path_center)
            cv2.line(output, (path_near, y_near), (path_far, y_far), (0, 255, 0), 5)
            cv2.circle(output, (path_far, y_far), 12, (0, 255, 0), -1)
            cv2.arrowedLine(
                output,
                (camera_x, int(height * 0.94)),
                (int(np.clip(path_far, 0, width - 1)), y_far),
                (255, 255, 255),
                10,
                tipLength=0.10,
            )

        cv2.line(
            output,
            (camera_x, int(height * 0.78)),
            (camera_x, int(height * 0.96)),
            (255, 100, 40),
            4,
        )

        if confidence < self.config.min_confidence:
            colour = (20, 20, 230)
        elif abs(angular_z) > 0.05:
            colour = (0, 165, 255)
        else:
            colour = (30, 200, 50)

        panel_height = max(138, int(height * 0.21))
        cv2.rectangle(output, (0, 0), (width, panel_height), (15, 20, 30), -1)
        scale = max(0.65, width / 1500.0)
        cv2.putText(output, instruction, (25, 42), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)
        cv2.putText(
            output,
            f"angular_z: {angular_z:+.3f} rad/s   linear_x: {linear_x:.3f} m/s",
            (25, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.72,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"lateral: {lateral_error:+.3f}   heading: {heading_error:+.3f}   confidence: {confidence:.2f}",
            (25, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.62,
            (210, 220, 230),
            2,
            cv2.LINE_AA,
        )
        if reference_target is None:
            reference_text = f"source: {control_source}"
        else:
            reference_text = (
                f"DB: {reference_target.filename}  idx={reference_target.match_index}  "
                f"VPR={reference_target.vpr_score:.1f}%  heading={reference_target.heading_deg}  "
                f"align={reference_alignment_error:+.3f}  turn={route_turn_error:+.3f}"
            )
        cv2.putText(
            output,
            reference_text,
            (25, 134),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.55,
            (100, 210, 255),
            2,
            cv2.LINE_AA,
        )
        return output
