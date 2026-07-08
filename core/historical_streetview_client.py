import csv
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class HistoricalPanorama:
    pano_id: str
    lat: float
    lon: float
    heading: float
    pitch: float | None
    roll: float | None
    date: str | None


def haversine_m(lat1, lon1, lat2, lon2):
    radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def bearing_deg(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    x = math.sin(delta_lon) * math.cos(phi2)
    y = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon)
    )
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def date_key(date_value):
    if not date_value:
        return (9999, 99)
    year, month = date_value.split("-")
    return (int(year), int(month))


def keep_by_date(date_value, before=None, after=None):
    if not date_value:
        return False
    current = date_key(date_value)
    if before is not None and current >= date_key(before):
        return False
    if after is not None and current < date_key(after):
        return False
    return True


def source_sort_key(source_index):
    try:
        return (int(source_index), source_index)
    except (TypeError, ValueError):
        return (10**9, str(source_index))


class HistoricalStreetViewClient:
    """Build a date-grouped historical Street View dataset from current route CSV."""

    def __init__(
        self,
        api_key,
        image_size="640x640",
        fov=100,
        pitch=0,
        before="2015-01",
        after=None,
        candidate_limit=3,
        sleep_s=0.2,
        request_timeout_s=30,
    ):
        self.api_key = api_key
        self.image_size = image_size
        self.fov = int(fov)
        self.pitch = int(pitch)
        self.before = before
        self.after = after
        self.candidate_limit = int(candidate_limit)
        self.sleep_s = float(sleep_s)
        self.request_timeout_s = int(request_timeout_s)

    def make_search_url(self, lat, lon):
        url = (
            "https://maps.googleapis.com/maps/api/js/"
            "GeoPhotoService.SingleImageSearch"
            "?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50"
            "!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10"
            "!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2"
            "&callback=callbackfunc"
        )
        return url.format(lat, lon)

    def search_panoramas(self, lat, lon):
        response = requests.get(
            self.make_search_url(lat, lon),
            timeout=self.request_timeout_s,
        )
        response.raise_for_status()

        matches = re.findall(r"callbackfunc\( (.*) \)$", response.text)
        if not matches:
            return []

        data = json.loads(matches[0])
        if data == [[5, "generic", "Search returned no images."]]:
            return []

        subset = data[1][5][0]
        raw_panos = subset[3][0][::-1]
        raw_dates = [] if (len(subset) < 9 or subset[8] is None) else subset[8][::-1]
        dates = [f"{item[1][0]}-{item[1][1]:02d}" for item in raw_dates]

        panoramas = []
        for index, pano in enumerate(raw_panos):
            pano_pose = pano[2][2]
            panoramas.append(
                HistoricalPanorama(
                    pano_id=pano[0][1],
                    lat=float(pano[2][0][2]),
                    lon=float(pano[2][0][3]),
                    heading=float(pano_pose[0]),
                    pitch=float(pano_pose[1]) if len(pano_pose) >= 2 else None,
                    roll=float(pano_pose[2]) if len(pano_pose) >= 3 else None,
                    date=dates[index] if index < len(dates) else None,
                )
            )
        return panoramas

    def build_route_headings(self, route_rows):
        coords_by_source = {}
        for row in route_rows:
            source_index = row.get("index", "")
            if not source_index or source_index in coords_by_source:
                continue
            coords_by_source[source_index] = (
                float(row["target_lat"]),
                float(row["target_lon"]),
            )

        ordered_sources = sorted(coords_by_source, key=source_sort_key)
        headings = {}
        for index, source_index in enumerate(ordered_sources):
            if len(ordered_sources) == 1:
                headings[source_index] = 0
                continue

            next_index = index + 1 if index < len(ordered_sources) - 1 else index
            prev_index = index - 1 if index == len(ordered_sources) - 1 else index
            start_source = ordered_sources[prev_index]
            end_source = ordered_sources[next_index]
            lat1, lon1 = coords_by_source[start_source]
            lat2, lon2 = coords_by_source[end_source]
            headings[source_index] = int(round(bearing_deg(lat1, lon1, lat2, lon2))) % 360
        return headings

    def download_static_image(self, pano_id, output_file, heading):
        url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            "size": self.image_size,
            "pano": pano_id,
            "heading": int(heading),
            "fov": self.fov,
            "pitch": self.pitch,
            "key": self.api_key,
        }
        response = requests.get(
            url,
            params=params,
            timeout=self.request_timeout_s,
        )
        response.raise_for_status()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("wb") as f:
            f.write(response.content)

    def clear_dynamic_dataset(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for child in output_dir.iterdir():
            if child.is_dir() and re.match(r"^\d{4}-\d{2}$", child.name):
                shutil.rmtree(child)
            elif child.name in {
                "historical_metadata.csv",
                "downloaded_static_metadata.csv",
            }:
                child.unlink()

    def search_historical_rows(self, route_rows, progress_callback=None):
        output_rows = []

        for source_rank, source in enumerate(route_rows):
            source_index = source.get("index", source_rank)
            lat = float(source["target_lat"])
            lon = float(source["target_lon"])

            if progress_callback is not None:
                progress_callback(
                    phase="search",
                    current=source_rank + 1,
                    total=len(route_rows),
                    downloaded=0,
                    skipped=0,
                    message=f"Searching historical panos for route point {source_index}",
                )

            try:
                panos = self.search_panoramas(lat, lon)
                panos = [
                    pano
                    for pano in panos
                    if keep_by_date(pano.date, self.before, self.after)
                ]
                panos = sorted(
                    panos,
                    key=lambda pano: (
                        haversine_m(lat, lon, pano.lat, pano.lon),
                        date_key(pano.date),
                        pano.pano_id,
                    ),
                )
                if self.candidate_limit > 0:
                    panos = panos[: self.candidate_limit]

                if not panos:
                    output_rows.append(
                        self.empty_historical_row(
                            source,
                            source_index,
                            lat,
                            lon,
                            "no_historical_pano",
                            "",
                        )
                    )
                else:
                    for candidate_rank, pano in enumerate(panos):
                        output_rows.append(
                            {
                                "source_index": source_index,
                                "source_lat": lat,
                                "source_lon": lon,
                                "source_current_pano_id": source.get("pano_id", ""),
                                "source_current_image_file": source.get("image_file", ""),
                                "candidate_rank": candidate_rank,
                                "historical_pano_id": pano.pano_id,
                                "historical_date": pano.date or "",
                                "historical_lat": pano.lat,
                                "historical_lon": pano.lon,
                                "distance_to_source_m": round(
                                    haversine_m(lat, lon, pano.lat, pano.lon),
                                    3,
                                ),
                                "heading": pano.heading,
                                "pitch": pano.pitch if pano.pitch is not None else "",
                                "roll": pano.roll if pano.roll is not None else "",
                                "search_status": "ok",
                                "error": "",
                            }
                        )
            except Exception as exc:
                output_rows.append(
                    self.empty_historical_row(
                        source,
                        source_index,
                        lat,
                        lon,
                        "error",
                        f"{type(exc).__name__}: {exc}",
                    )
                )

            time.sleep(self.sleep_s)

        return output_rows

    def empty_historical_row(self, source, source_index, lat, lon, status, error):
        return {
            "source_index": source_index,
            "source_lat": lat,
            "source_lon": lon,
            "source_current_pano_id": source.get("pano_id", ""),
            "source_current_image_file": source.get("image_file", ""),
            "candidate_rank": "",
            "historical_pano_id": "",
            "historical_date": "",
            "historical_lat": "",
            "historical_lon": "",
            "distance_to_source_m": "",
            "heading": "",
            "pitch": "",
            "roll": "",
            "search_status": status,
            "error": error,
        }

    def write_rows(self, csv_file, rows, fieldnames):
        csv_file = Path(csv_file)
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        with csv_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def acquire_from_current_metadata(
        self,
        current_metadata_file,
        output_dir,
        progress_callback=None,
    ):
        current_metadata_file = Path(current_metadata_file)
        output_dir = Path(output_dir)

        with current_metadata_file.open(newline="", encoding="utf-8") as f:
            route_rows = list(csv.DictReader(f))

        if not route_rows:
            raise ValueError("Current Street View metadata CSV is empty.")

        self.clear_dynamic_dataset(output_dir)

        historical_rows = self.search_historical_rows(
            route_rows,
            progress_callback=progress_callback,
        )
        historical_fieldnames = [
            "source_index",
            "source_lat",
            "source_lon",
            "source_current_pano_id",
            "source_current_image_file",
            "candidate_rank",
            "historical_pano_id",
            "historical_date",
            "historical_lat",
            "historical_lon",
            "distance_to_source_m",
            "heading",
            "pitch",
            "roll",
            "search_status",
            "error",
        ]
        self.write_rows(
            output_dir / "historical_metadata.csv",
            historical_rows,
            historical_fieldnames,
        )

        downloadable_rows = [
            row
            for row in historical_rows
            if row.get("search_status") == "ok" and row.get("historical_pano_id")
        ]
        route_headings = self.build_route_headings(route_rows)
        output_rows = []
        downloaded = 0
        skipped = 0

        for index, row in enumerate(downloadable_rows):
            source_index = str(row["source_index"])
            candidate_rank = str(row["candidate_rank"])
            pano_id = str(row["historical_pano_id"])
            date = row.get("historical_date") or "unknown"
            heading = route_headings.get(source_index, 0)
            filename = (
                f"source_{int(source_index):04d}_rank_{candidate_rank}_{date}_"
                f"heading_{heading:03d}_{pano_id}.jpg"
            )
            image_path = output_dir / date / filename

            if progress_callback is not None:
                progress_callback(
                    phase="download",
                    current=index + 1,
                    total=len(downloadable_rows),
                    downloaded=downloaded,
                    skipped=skipped,
                    message=(
                        f"Downloading historical image {index + 1}/"
                        f"{len(downloadable_rows)} ({date})"
                    ),
                )

            status = "ok"
            error = ""
            try:
                self.download_static_image(
                    pano_id=pano_id,
                    output_file=image_path,
                    heading=heading,
                )
                downloaded += 1
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                skipped += 1

            output_row = dict(row)
            output_row["static_image_path"] = str(image_path) if status == "ok" else ""
            output_row["static_heading"] = str(heading)
            output_row["heading_source"] = "route"
            output_row["download_status"] = status
            output_row["download_error"] = error
            output_rows.append(output_row)
            time.sleep(self.sleep_s)

        download_fieldnames = historical_fieldnames + [
            "static_image_path",
            "static_heading",
            "heading_source",
            "download_status",
            "download_error",
        ]
        self.write_rows(
            output_dir / "downloaded_static_metadata.csv",
            output_rows,
            download_fieldnames,
        )

        if progress_callback is not None:
            progress_callback(
                phase="done",
                current=len(downloadable_rows),
                total=len(downloadable_rows),
                downloaded=downloaded,
                skipped=skipped,
                message="Historical Street View acquisition completed.",
            )

        dates = sorted(
            {
                row.get("historical_date")
                for row in output_rows
                if row.get("download_status") == "ok" and row.get("historical_date")
            }
        )
        return {
            "route_points": len(route_rows),
            "historical_candidates": len(downloadable_rows),
            "downloaded": downloaded,
            "skipped": skipped,
            "dates": dates,
            "metadata_file": str(output_dir / "downloaded_static_metadata.csv"),
        }
