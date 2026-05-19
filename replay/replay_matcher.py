import os
import csv


class ReplayMatcher:
    def __init__(
        self,
        metadata_file,
    ):
        self.metadata_file = metadata_file
        self.metadata = self.load_metadata()

    def load_metadata(self):
        metadata = {}

        if not os.path.exists(
            self.metadata_file
        ):
            return metadata

        with open(
            self.metadata_file,
            "r",
            encoding="utf-8"
        ) as f:
            reader = csv.DictReader(f)

            for row in reader:
                image_file = row.get(
                    "image_file",
                    ""
                )

                image_name = os.path.basename(
                    image_file
                )

                metadata[image_name] = row

        return metadata

    def attach_gps(self, result):
        filename = result.get("filename")

        meta = self.metadata.get(
            filename
        )

        if meta is None:
            result["lat"] = None
            result["lon"] = None
            return result

        result["lat"] = float(
            meta["target_lat"]
        )

        result["lon"] = float(
            meta["target_lon"]
        )

        return result

    @staticmethod
    def save_results(
        results,
        output_file,
    ):
        os.makedirs(
            os.path.dirname(output_file),
            exist_ok=True
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame",
                    "best_match",
                    "score",
                    "lat",
                    "lon",
                    "match_index",
                ]
            )

            writer.writeheader()

            for item in results:
                writer.writerow(item)