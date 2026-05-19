import os
import time

from replay.replay_matcher import (
    ReplayMatcher
)

from matching.orb_matcher import (
    ORBMatcher
)

from matching.cnn_matcher import (
    CNNMatcher
)

from matching.hybrid_matcher import (
    HybridMatcher
)


class ReplayLocalizer:

    def __init__(
        self,
        database_dir,
        metadata_file,
        matching_mode="Hybrid",
        top_k=5,
        orb_features=3000,
        orb_distance_threshold=55,
        cnn_image_size=224,
        orb_weight=0.4,
        cnn_weight=0.6,
        search_window=3,
        max_jump=3,
        min_score=40,
    ):

        self.database_dir = database_dir
        self.metadata_file = metadata_file
        self.matching_mode = matching_mode
        self.top_k = top_k

        self.search_window = search_window
        self.max_jump = max_jump
        self.min_score = min_score

        self.last_match_index = None

        self.replay_matcher = ReplayMatcher(
            metadata_file
        )

        self.orb = ORBMatcher(
            n_features=orb_features,
            distance_threshold=orb_distance_threshold,
        )

        self.cnn = CNNMatcher(
            image_size=cnn_image_size
        )

        self.hybrid = HybridMatcher(
            orb_weight=orb_weight,
            cnn_weight=cnn_weight,
        )

    # =====================================================
    # GET DATABASE IMAGES
    # =====================================================

    def get_database_images(self):

        return sorted(
            file for file in os.listdir(
                self.database_dir
            )
            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )
        )

    # =====================================================
    # GET SEARCH WINDOW
    # =====================================================

    def get_candidate_images(
        self,
        database_images,
    ):

        if self.last_match_index is None:
            return database_images

        start_index = max(
            0,
            self.last_match_index
            - self.search_window
        )

        end_index = min(
            len(database_images),
            self.last_match_index
            + self.search_window
            + 1
        )

        return database_images[
            start_index:end_index
        ]

    # =====================================================
    # LOCAL SEARCH
    # =====================================================

    def local_search(
        self,
        frame_path,
        candidate_images,
    ):

        results = []

        for filename in candidate_images:

            image_path = os.path.join(
                self.database_dir,
                filename
            )

            if self.matching_mode == "ORB":

                comparison = self.orb.compare_images(
                    frame_path,
                    image_path
                )

                result = {
                    "filename": filename,
                    "score": comparison["score"],
                    "good_matches": comparison["good_matches"],
                }

            elif self.matching_mode == "CNN":

                comparison = self.cnn.compare_images(
                    frame_path,
                    image_path
                )

                result = {
                    "filename": filename,
                    "score": comparison["score"],
                }

            else:

                orb_comparison = self.orb.compare_images(
                    frame_path,
                    image_path
                )

                cnn_comparison = self.cnn.compare_images(
                    frame_path,
                    image_path
                )

                fused_score = (
                    self.hybrid.compute_hybrid_score(
                        orb_score=orb_comparison["score"],
                        cnn_score=cnn_comparison["score"],
                    )
                )

                result = {
                    "filename": filename,
                    "score": fused_score,
                    "orb_score": orb_comparison["score"],
                    "cnn_score": cnn_comparison["score"],
                    "good_matches": orb_comparison["good_matches"],
                }

            results.append(result)

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:self.top_k]

    # =====================================================
    # CANDIDATE VALIDATION
    # =====================================================

    def validate_candidate(
        self,
        candidate,
        database_images,
    ):

        if candidate["score"] < self.min_score:
            return None

        filename = candidate.get("filename")

        if filename not in database_images:
            return None

        current_index = database_images.index(
            filename
        )

        if self.last_match_index is not None:

            jump = abs(
                current_index
                - self.last_match_index
            )

            if jump > self.max_jump:
                return None

        return current_index

    # =====================================================
    # LOCALIZE FRAME
    # =====================================================

    def localize_frame(
        self,
        frame_path,
    ):

        database_images = self.get_database_images()

        candidate_images = self.get_candidate_images(
            database_images
        )

        print(
            f"Search window: "
            f"{len(candidate_images)} images"
        )

        results = self.local_search(
            frame_path,
            candidate_images,
        )

        if not results:
            return None

        best = None
        best_index = None

        for candidate in results:

            current_index = self.validate_candidate(
                candidate,
                database_images
            )

            if current_index is None:
                continue

            best = candidate
            best_index = current_index
            break

        if best is None:
            return None

        best = self.replay_matcher.attach_gps(
            best
        )

        self.last_match_index = best_index

        return best

    # =====================================================
    # FULL REPLAY LOCALIZATION
    # =====================================================

    def localize_frames(
        self,
        frames_dir,
        progress_callback=None,
    ):

        frame_files = sorted(
            file for file in os.listdir(
                frames_dir
            )
            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )
        )

        replay_results = []

        total = len(frame_files)

        start_time = time.time()

        for i, filename in enumerate(
            frame_files
        ):

            frame_path = os.path.join(
                frames_dir,
                filename
            )

            best = self.localize_frame(
                frame_path
            )

            if best is not None:

                replay_results.append({
                    "frame": filename,
                    "best_match": best.get("filename"),
                    "score": best.get("score"),
                    "lat": best.get("lat"),
                    "lon": best.get("lon"),
                    "match_index": self.last_match_index,
                })

            if progress_callback is not None:

                elapsed = time.time() - start_time

                remaining = 0

                if i > 0:
                    estimated_total = (
                        elapsed / (i + 1)
                    ) * total

                    remaining = (
                        estimated_total
                        - elapsed
                    )

                progress_callback(
                    current=i + 1,
                    total=total,
                    remaining=remaining,
                    filename=filename,
                )

        return replay_results