from __future__ import annotations

from dataclasses import dataclass
import csv
import logging
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageRecord:
    id: str
    path: Path
    latitude: float | None = None
    longitude: float | None = None
    heading: float | None = None
    timestamp: str | None = None
    route_index: int | None = None
    target_latitude: float | None = None
    target_longitude: float | None = None
    pano_id: str | None = None
    pano_latitude: float | None = None
    pano_longitude: float | None = None
    distance_to_pano_m: float | None = None

    @property
    def has_position(self) -> bool:
        return self.latitude is not None and self.longitude is not None


@dataclass(frozen=True)
class ImageCollection:
    name: str
    records: tuple[ImageRecord, ...]

    @classmethod
    def from_directory(
        cls,
        name: str,
        directory: str | Path,
        metadata_csv: str | Path | None = None,
    ) -> "ImageCollection":
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {root}")
        paths = sorted(
            (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda p: p.name.lower(),
        )
        if not paths:
            raise ValueError(f"No supported images found in {root}")
        metadata_path = Path(metadata_csv) if metadata_csv else _discover_metadata_csv(root, paths)
        if metadata_csv and not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_path}")
        sidecar = _read_sidecar(metadata_path)
        records = []
        for path in paths:
            metadata = _read_exif(path)
            metadata.update(sidecar.get(path.name, sidecar.get(path.stem, {})))
            records.append(ImageRecord(path.stem, path.resolve(), **metadata))
        if records and all(record.route_index is not None for record in records):
            records.sort(key=lambda record: record.route_index)
        matched = sum(record.route_index is not None or record.has_position for record in records)
        if sidecar:
            LOGGER.info("Loaded metadata for %d/%d images from %s", matched, len(records), metadata_path)
        return cls(name=name, records=tuple(records))

    def __len__(self) -> int:
        return len(self.records)

    def subset(self, indices: list[int] | tuple[int, ...] | Any, name: str | None = None) -> "ImageCollection":
        selected = tuple(self.records[int(index)] for index in indices)
        if not selected:
            raise ValueError("An image collection subset cannot be empty")
        return ImageCollection(name=name or self.name, records=selected)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_sidecar(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image_value = row.get("filename") or row.get("id") or row.get("image") or row.get("image_file")
            key = Path(image_value).name if image_value else None
            if not key:
                continue
            target_latitude = _optional_float(row.get("target_lat") or row.get("latitude") or row.get("lat"))
            target_longitude = _optional_float(
                row.get("target_lon") or row.get("longitude") or row.get("lon") or row.get("lng")
            )
            values = {
                # The route target is the best estimate of where the robot should be.
                "latitude": target_latitude,
                "longitude": target_longitude,
                "heading": _optional_float(row.get("heading") or row.get("bearing")),
                "timestamp": row.get("timestamp") or row.get("datetime") or None,
                "route_index": _optional_int(row.get("index")),
                "target_latitude": target_latitude,
                "target_longitude": target_longitude,
                "pano_id": row.get("pano_id") or None,
                "pano_latitude": _optional_float(row.get("pano_lat")),
                "pano_longitude": _optional_float(row.get("pano_lon")),
                "distance_to_pano_m": _optional_float(row.get("distance_to_pano_m")),
            }
            result[key] = values
            result[Path(key).stem] = values
    return result


def _discover_metadata_csv(root: Path, image_paths: list[Path]) -> Path:
    direct = root / "metadata.csv"
    if direct.is_file():
        return direct
    candidates = sorted({*root.glob("*.csv"), *root.parent.glob("*.csv")})
    image_names = {path.name for path in image_paths}
    scored: list[tuple[int, Path]] = []
    for candidate in candidates:
        metadata = _read_sidecar(candidate)
        matches = sum(name in metadata for name in image_names)
        if matches:
            scored.append((matches, candidate))
    if not scored:
        return direct
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    best_count = scored[0][0]
    best = [path for count, path in scored if count == best_count]
    if len(best) == 1:
        LOGGER.info("Automatically selected Street View metadata CSV: %s", best[0])
        return best[0]
    LOGGER.warning(
        "Several metadata CSV files match %d images; set metadata_csv explicitly: %s",
        best_count,
        ", ".join(str(path) for path in best),
    )
    return direct


def _gps_decimal(value: Any, reference: str | None) -> float | None:
    if not value or len(value) != 3:
        return None
    degrees, minutes, seconds = (float(part) for part in value)
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    return -decimal if reference in {"S", "W"} else decimal


def _read_exif(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            if not exif:
                return metadata
            timestamp = exif.get(36867) or exif.get(306)
            if timestamp:
                metadata["timestamp"] = str(timestamp)
            # 34853 is the standard EXIF GPSInfo IFD tag and works across Pillow versions.
            gps_ifd = exif.get_ifd(34853)
            if gps_ifd:
                latitude = _gps_decimal(gps_ifd.get(2), gps_ifd.get(1))
                longitude = _gps_decimal(gps_ifd.get(4), gps_ifd.get(3))
                if latitude is not None and longitude is not None:
                    metadata["latitude"] = latitude
                    metadata["longitude"] = longitude
                heading = _optional_float(gps_ifd.get(17))
                if heading is not None:
                    metadata["heading"] = heading
    except (OSError, ValueError, TypeError, KeyError):
        return metadata
    return metadata
