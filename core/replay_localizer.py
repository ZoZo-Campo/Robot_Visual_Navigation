import os
import time

from replay.replay_matcher import ReplayMatcher

from matching.cnn_matcher import CNNMatcher
from matching.dino_matcher import DINOMatcher


class ReplayLocalizer:

    def __init__(
        self,
        database_dir,
        metadata_file,
        matching_mode="Hybrid",
        top_k=5,
        cnn_image_size=224,
        cnn_weight=0.45,
        dino_weight=0.55,
        search_window=12,
        max_forward_jump=12,
        max_backward_jump=3,
        min_score=35,
    ):
        self.database_dir = database_dir
        self.metadata_file = metadata_file
        self.matching_mode = matching_mode
        self.top_k = top_k

        self.cnn_weight = cnn_weight
        self.dino_weight = dino_weight

        self.search_window = search_window
        self.max_forward_jump = max_forward_jump
        self.max_backward_jump = max_backward_jump
        self.min_score = min_score

        self.last_match_index = None

        self.replay_matcher = ReplayMatcher(
            metadata_file
        )

        self.cnn = CNNMatcher(
            image_size=cnn_image_size
        )

        self.dino = DINOMatcher()

    # =====================================================
    # DATABASE
    # =====================================================

    def get_database_images(self):
        return sorted(
            file for file in os.listdir(self.database_dir)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        )

    # =====================================================
    # LOCAL WINDOW AROUND PREVIOUS MATCH
    # =====================================================

    def get_candidate_images(
        self,
        database_images,
    ):
        if self.last_match_index is None:
            return database_images

        start_index = max(
            0,
            self.last_match_index - self.search_window
        )

        end_index = min(
            len(database_images),
            self.last_match_index + self.search_window + 1
        )

        return database_images[start_index:end_index]

    # =====================================================
    # HYBRID CNN + DINO SCORE
    # =====================================================

    def compute_hybrid_score(
        self,
        cnn_score,
        dino_score,
    ):
        return (
            self.cnn_weight * cnn_score
            +
            self.dino_weight * dino_score
        )

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

            if self.matching_mode == "CNN":

                cnn_result = self.cnn.compare_images(
                    frame_path,
                    image_path
                )

                result = {
                    "filename": filename,
                    "score": cnn_result["score"],
                    "cnn_score": cnn_result["score"],
                    "dino_score": None,
                }

            elif self.matching_mode == "DINO":

                dino_result = self.dino.compare_images(
                    frame_path,
                    image_path
                )

                result = {
                    "filename": filename,
                    "score": dino_result["score"],
                    "cnn_score": None,
                    "dino_score": dino_result["score"],
                }

            else:

                cnn_result = self.cnn.compare_images(
                    frame_path,
                    image_path
                )

                dino_result = self.dino.compare_images(
                    frame_path,
                    image_path
                )

                score = self.compute_hybrid_score(
                    cnn_score=cnn_result["score"],
                    dino_score=dino_result["score"],
                )

                result = {
                    "filename": filename,
                    "score": score,
                    "cnn_score": cnn_result["score"],
                    "dino_score": dino_result["score"],
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

            delta = current_index - self.last_match_index

            if delta > self.max_forward_jump:
                return None

            if delta < -self.max_backward_jump:
                return None

        return current_index

    # =====================================================
    # LOCALIZE ONE FRAME
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
            f"Search window: {len(candidate_images)} images"
        )

        results = self.local_search(
            frame_path,
            candidate_images
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

        best["match_index"] = best_index

        return best

    # =====================================================
    # LOCALIZE ALL FRAMES
    # =====================================================

    def localize_frames(
        self,
        frames_dir,
        progress_callback=None,
    ):
        frame_files = sorted(
            file for file in os.listdir(frames_dir)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        )

        replay_results = []

        total = len(frame_files)
        start_time = time.time()

        for i, filename in enumerate(frame_files):

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
                    "cnn_score": best.get("cnn_score"),
                    "dino_score": best.get("dino_score"),
                    "lat": best.get("lat"),
                    "lon": best.get("lon"),
                    "match_index": best.get("match_index"),
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