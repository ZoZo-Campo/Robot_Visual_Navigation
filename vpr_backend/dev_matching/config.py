from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetPair:
    robot_name: str
    robot_dir: Path
    streetview_name: str
    streetview_dir: Path
    robot_metadata_csv: Path | None = None
    streetview_metadata_csv: Path | None = None


@dataclass(frozen=True)
class MethodConfig:
    name: str
    weight: float = 1.0
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryConfig:
    algorithm: str = "local_window_search"
    max_step: int = 3
    stay_penalty: float = 0.15
    jump_penalty: float = 0.4
    backward_penalty: float = 1.0
    emission_temperature: float = 0.1
    progress_weight: float = 0.28
    progress_sigma_fraction: float = 0.20
    speed_penalty: float = 0.18
    initial_search_size: int = 5
    initial_evidence_frames: int = 3
    initial_index_penalty: float = 0.04
    window_backward: int = 0
    window_forward: int = 5
    max_window_distance_m: float = 30.0
    temporal_smoothing: float = 0.65
    local_stay_penalty: float = 0.01


@dataclass(frozen=True)
class GeometryConfig:
    enabled: bool = True
    score_weight: float = 0.12
    heading_sigma_degrees: float = 35.0
    pano_distance_scale_m: float = 12.0
    route_heading_break_degrees: float = 100.0
    restrict_to_primary_heading_segment: bool = True


@dataclass(frozen=True)
class FilteringConfig:
    enabled: bool = True
    method: str = "rosa_v1"
    descriptor_method: str = "spatial_pyramid"
    enforce_dominant_direction: bool = True
    deduplicate_panoramas: bool = True
    min_spacing_m: float = 4.0
    max_spacing_m: float = 15.0
    visual_similarity_threshold: float = 0.95
    sequence_outlier_neighbor_max: float = 0.78
    sequence_outlier_bridge_min: float = 0.85
    preserve_unfiltered_matrix: bool = True


@dataclass(frozen=True)
class AppConfig:
    pairs: tuple[DatasetPair, ...]
    methods: tuple[MethodConfig, ...]
    cache_dir: Path = Path("cache")
    output_dir: Path = Path("outputs")
    batch_size: int = 16
    device: str = "auto"
    fusion: str = "rank"
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _dataset_source(base: Path, spec: str | dict[str, Any]) -> tuple[Path, Path | None]:
    if isinstance(spec, str):
        return _resolve(base, spec), None
    if not isinstance(spec, dict):
        raise TypeError("A dataset must be a path or a mapping with 'path' and optional 'metadata_csv'")
    directory = spec.get("path") or spec.get("directory")
    if not directory:
        raise ValueError("Dataset mapping is missing 'path'")
    metadata = spec.get("metadata_csv")
    return _resolve(base, directory), _resolve(base, metadata) if metadata else None


def load_config(path: str | Path) -> AppConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML configuration requires PyYAML: pip install PyYAML") from exc
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    datasets = raw.get("datasets", {})
    robot = datasets.get("robot", {})
    streetview = datasets.get("streetview", {})
    pairs_list = []
    for r_name, r_spec in robot.items():
        robot_dir, robot_metadata = _dataset_source(base, r_spec)
        for s_name, s_spec in streetview.items():
            streetview_dir, streetview_metadata = _dataset_source(base, s_spec)
            pairs_list.append(
                DatasetPair(
                    robot_name=r_name,
                    robot_dir=robot_dir,
                    streetview_name=s_name,
                    streetview_dir=streetview_dir,
                    robot_metadata_csv=robot_metadata,
                    streetview_metadata_csv=streetview_metadata,
                )
            )
    pairs = tuple(pairs_list)
    method_configs = []
    for name, spec in raw.get("methods", {}).items():
        params = dict(spec.get("params", {}))
        if params.get("checkpoint"):
            params["checkpoint"] = str(_resolve(base, params["checkpoint"]))
        method_configs.append(
            MethodConfig(
                name=name,
                enabled=bool(spec.get("enabled", True)),
                weight=float(spec.get("weight", 1.0)),
                params=params,
            )
        )
    methods = tuple(method_configs)
    trajectory = TrajectoryConfig(**raw.get("trajectory", {}))
    geometry = GeometryConfig(**raw.get("geometry", {}))
    filtering = FilteringConfig(**raw.get("filtering", {}))
    return AppConfig(
        pairs=pairs,
        methods=methods,
        cache_dir=_resolve(base, raw.get("cache_dir", "../cache")),
        output_dir=_resolve(base, raw.get("output_dir", "../outputs")),
        batch_size=int(raw.get("batch_size", 16)),
        device=str(raw.get("device", "auto")),
        fusion=str(raw.get("fusion", "rank")),
        geometry=geometry,
        filtering=filtering,
        trajectory=trajectory,
    )
