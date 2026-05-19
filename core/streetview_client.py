import os
import time
import requests

from core.gps_utils import (
    haversine_m,
    calculate_heading,
)


class StreetViewClient:

    def __init__(
        self,
        api_key,
        image_size="640x640",
        fov=100,
        pitch=-5,
        radius=8,
        max_distance_to_pano=8,
        heading_mode="route",
    ):
        self.api_key = api_key
        self.image_size = image_size
        self.fov = fov
        self.pitch = pitch
        self.radius = radius
        self.max_distance_to_pano = max_distance_to_pano
        self.heading_mode = heading_mode

    def get_metadata(self, lat, lon):
        url = "https://maps.googleapis.com/maps/api/streetview/metadata"

        params = {
            "location": f"{lat},{lon}",
            "radius": self.radius,
            "source": "outdoor",
            "key": self.api_key,
        }

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        return response.json()

    def download_image(self, pano_id, filename, heading):
        url = "https://maps.googleapis.com/maps/api/streetview"

        params = {
            "size": self.image_size,
            "pano": pano_id,
            "heading": heading,
            "fov": self.fov,
            "pitch": self.pitch,
            "key": self.api_key,
        }

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)

            return True

        return False

    def compute_heading(self, route, index):
        if self.heading_mode == "north":
            return 0

        if index < len(route) - 1:
            return calculate_heading(
                route[index],
                route[index + 1]
            )

        return calculate_heading(
            route[index - 1],
            route[index]
        )

    def acquire_from_route(
        self,
        route,
        output_dir,
        metadata_file,
        progress_callback=None,
    ):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)

        total = len(route)
        downloaded = 0
        skipped = 0

        with open(metadata_file, "w", encoding="utf-8") as meta:
            meta.write(
                "index,target_lat,target_lon,pano_id,pano_lat,pano_lon,"
                "distance_to_pano_m,heading,image_file\n"
            )

            for i, (lat, lon) in enumerate(route):

                if progress_callback is not None:
                    progress_callback(
                        current=i + 1,
                        total=total,
                        downloaded=downloaded,
                        skipped=skipped,
                        message=f"Processing point {i + 1}/{total}",
                    )

                metadata = self.get_metadata(lat, lon)

                if metadata.get("status") != "OK":
                    skipped += 1
                    continue

                pano_id = metadata["pano_id"]
                pano_lat = metadata["location"]["lat"]
                pano_lon = metadata["location"]["lng"]

                distance = haversine_m(
                    (lat, lon),
                    (pano_lat, pano_lon)
                )

                if distance > self.max_distance_to_pano:
                    skipped += 1
                    continue

                heading = self.compute_heading(route, i)

                filename = os.path.join(
                    output_dir,
                    f"streetview_{i:04d}.jpg"
                )

                ok = self.download_image(
                    pano_id=pano_id,
                    filename=filename,
                    heading=heading,
                )

                if ok:
                    downloaded += 1

                    meta.write(
                        f"{i},{lat},{lon},{pano_id},{pano_lat},{pano_lon},"
                        f"{distance:.2f},{heading:.2f},{filename}\n"
                    )

                else:
                    skipped += 1

                time.sleep(0.2)

        if progress_callback is not None:
            progress_callback(
                current=total,
                total=total,
                downloaded=downloaded,
                skipped=skipped,
                message="Street View acquisition completed.",
            )

        return {
            "total": total,
            "downloaded": downloaded,
            "skipped": skipped,
        }