from __future__ import annotations

from pathlib import Path

from config import VPR_OUTPUT_DIR
from matching.vpr_matcher import VPRMatcher


class ReplayLocalizer:
    """Sequence localizer backed by cached VPR descriptors and local trajectory search."""

    def __init__(
        self,
        database_dir,
        metadata_file,
        matching_mode="MixVPR",
        top_k=5,
        min_score=0,
        output_dir=VPR_OUTPUT_DIR,
        **_,
    ):
        self.top_k = top_k
        self.min_score = min_score
        self.output_dir = Path(output_dir)
        self.matcher = VPRMatcher(
            database_dir=database_dir,
            metadata_file=metadata_file,
            mode=matching_mode,
        )

    @property
    def database_summary(self):
        return self.matcher.summary

    def localize_frame(self, frame_path):
        results = self.matcher.search(frame_path, top_k=self.top_k)
        return results[0] if results else None

    def localize_frames(self, frames_dir, progress_callback=None):
        return self.matcher.localize_sequence(
            frames_dir=frames_dir,
            top_k=self.top_k,
            min_score=self.min_score,
            output_dir=self.output_dir,
            progress_callback=progress_callback,
        )
