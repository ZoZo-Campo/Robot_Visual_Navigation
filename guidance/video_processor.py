"""Offline video simulation for the V4 visual guidance controller."""

import csv
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .path_follower import GuidanceConfig, VisualPathFollower


class GuidanceVideoProcessor:
    CSV_FIELDS = [
        "frame_index",
        "timestamp_s",
        "camera_center_x_px",
        "path_center_x_px",
        "lateral_error",
        "heading_error",
        "angular_z_rad_s",
        "linear_x_m_s",
        "confidence",
        "instruction",
        "left_boundary_detected",
        "right_boundary_detected",
        "control_source",
        "reference_filename",
        "reference_match_index",
        "reference_latitude",
        "reference_longitude",
        "reference_heading_deg",
        "reference_vpr_score",
        "reference_alignment_error",
        "route_turn_error",
        "reference_confidence",
    ]

    def __init__(self, config=None):
        self.config = config or GuidanceConfig()

    @staticmethod
    def _compress_for_browser(raw_video_path, output_video_path):
        """Create a compact H.264 MP4 when ffmpeg is available."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            os.replace(raw_video_path, output_video_path)
            return "mpeg4"

        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw_video_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_video_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError):
            os.replace(raw_video_path, output_video_path)
            return "mpeg4"

        os.remove(raw_video_path)
        return "h264"

    def process_video(
        self,
        video_path,
        output_video_path,
        output_csv_path,
        processing_fps=10.0,
        reference_targets=None,
        reference_interval_s=2.0,
        progress_callback=None,
    ):
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        input_fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if input_fps <= 0 or width <= 0 or height <= 0:
            capture.release()
            raise ValueError("Invalid video metadata.")

        output_fps = max(1.0, min(float(processing_fps), input_fps))
        frame_step = max(1, int(round(input_fps / output_fps)))
        effective_fps = input_fps / frame_step

        os.makedirs(os.path.dirname(str(output_video_path)), exist_ok=True)
        os.makedirs(os.path.dirname(str(output_csv_path)), exist_ok=True)
        output_video_path = Path(output_video_path)
        raw_video_path = output_video_path.with_name(
            f"{output_video_path.stem}_raw{output_video_path.suffix}"
        )
        writer = cv2.VideoWriter(
            str(raw_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            effective_fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError(f"Cannot create output video: {raw_video_path}")

        follower = VisualPathFollower(self.config)
        rows = []
        frame_index = 0
        processed = 0
        confidence_values = []
        angular_values = []
        lost_count = 0
        db_guided_count = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % frame_step == 0:
                    timestamp_s = frame_index / input_fps
                    reference_target = None
                    if reference_targets:
                        target_index = min(
                            int(timestamp_s / max(float(reference_interval_s), 1e-6)),
                            len(reference_targets) - 1,
                        )
                        reference_target = reference_targets[target_index]
                    result = follower.process(
                        frame,
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        reference_target=reference_target,
                    )
                    writer.write(result.annotated_frame)
                    rows.append(result.as_csv_row())
                    confidence_values.append(result.confidence)
                    angular_values.append(result.angular_z)
                    lost_count += int(result.instruction.startswith("STOP"))
                    db_guided_count += int(result.control_source == "STREETVIEW_DB")
                    processed += 1

                frame_index += 1
                if progress_callback is not None and (
                    frame_index % max(frame_step * 5, 1) == 0
                    or frame_index >= total_frames
                ):
                    progress_callback(
                        current=min(frame_index, total_frames),
                        total=total_frames,
                        processed=processed,
                    )
        finally:
            capture.release()
            writer.release()

        video_codec = self._compress_for_browser(
            raw_video_path,
            output_video_path,
        )

        with open(output_csv_path, "w", encoding="utf-8", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.CSV_FIELDS)
            csv_writer.writeheader()
            csv_writer.writerows(rows)

        confidence_array = np.asarray(confidence_values, dtype=np.float32)
        angular_array = np.asarray(angular_values, dtype=np.float32)
        return {
            "input_video": str(video_path),
            "output_video": str(output_video_path),
            "output_csv": str(output_csv_path),
            "input_fps": input_fps,
            "processing_fps": effective_fps,
            "total_input_frames": total_frames,
            "processed_frames": processed,
            "mean_confidence": float(confidence_array.mean()) if processed else 0.0,
            "mean_abs_angular_z": float(np.abs(angular_array).mean()) if processed else 0.0,
            "max_abs_angular_z": float(np.abs(angular_array).max()) if processed else 0.0,
            "path_lost_frames": lost_count,
            "path_lost_ratio": float(lost_count / processed) if processed else 1.0,
            "db_guided_frames": db_guided_count,
            "db_guided_ratio": float(db_guided_count / processed) if processed else 0.0,
            "output_codec": video_codec,
        }
