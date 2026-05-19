import os
import cv2


class ORBMatcher:

    def __init__(
        self,
        n_features=3000,
        distance_threshold=55,
    ):
        self.n_features = n_features
        self.distance_threshold = distance_threshold

        self.orb = cv2.ORB_create( # type: ignore
            nfeatures=self.n_features
        )

        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=True
        )

    def load_image_gray(self, image_path):
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(
                f"Cannot read image: {image_path}"
            )

        return image

    def extract_features(self, image_path):
        image = self.load_image_gray(image_path)

        keypoints, descriptors = self.orb.detectAndCompute(
            image,
            None
        )

        return keypoints, descriptors

    def compare_images(
        self,
        query_image_path,
        database_image_path,
    ):
        query_kp, query_desc = self.extract_features(
            query_image_path
        )

        db_kp, db_desc = self.extract_features(
            database_image_path
        )

        if (
            query_desc is None
            or db_desc is None
        ):
            return {
                "score": 0.0,
                "good_matches": 0,
                "query_keypoints": len(query_kp),
                "database_keypoints": len(db_kp),
            }

        matches = self.matcher.match(
            query_desc,
            db_desc
        )

        good_matches = [
            match for match in matches
            if match.distance < self.distance_threshold
        ]

        max_possible_matches = max(
            1,
            min(
                len(query_kp),
                len(db_kp)
            )
        )

        score = (
            len(good_matches)
            / max_possible_matches
        ) * 100

        return {
            "score": score,
            "good_matches": len(good_matches),
            "query_keypoints": len(query_kp),
            "database_keypoints": len(db_kp),
        }

    def search(
        self,
        query_image_path,
        database_dir,
        top_k=5,
        progress_callback=None,
    ):
        image_files = sorted(
            file for file in os.listdir(database_dir)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        )

        results = []

        total = len(image_files)

        for i, filename in enumerate(image_files):
            image_path = os.path.join(
                database_dir,
                filename
            )

            comparison = self.compare_images(
                query_image_path,
                image_path
            )

            result = {
                "filename": filename,
                "score": comparison["score"],
                "good_matches": comparison["good_matches"],
                "query_keypoints": comparison["query_keypoints"],
                "database_keypoints": comparison["database_keypoints"],
            }

            results.append(result)

            if progress_callback is not None:
                progress_callback(
                    current=i + 1,
                    total=total,
                    result=result,
                )

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]