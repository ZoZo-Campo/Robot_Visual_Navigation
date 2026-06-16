from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from dev_matching.data import ImageCollection


def _thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#20242a")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def write_match_sheets(
    output_dir: str | Path,
    robot: ImageCollection,
    streetview: ImageCollection,
    scores: np.ndarray,
    trajectory: np.ndarray,
    top_k: int = 5,
    window_starts: np.ndarray | None = None,
    window_stops: np.ndarray | None = None,
) -> None:
    """Create one readable robot-versus-Street-View PNG per query frame."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for query_index in range(len(robot)):
        start = int(window_starts[query_index]) if window_starts is not None else 0
        stop = int(window_stops[query_index]) if window_stops is not None else len(streetview) - 1
        candidates = np.arange(start, stop + 1)
        ranked_indices = candidates[np.argsort(-scores[query_index, candidates])[: min(top_k, len(candidates))]]
        query = robot.records[query_index]
        sheet = Image.new("RGB", (1500, 920), "#111418")
        draw = ImageDraw.Draw(sheet)
        draw.text((35, 25), f"Robot: {query.id} | fichier: {query.path.name}", fill="white", font=font)
        sheet.paste(_thumbnail(query.path, (650, 760)), (35, 70))
        draw.rectangle((34, 69, 686, 831), outline="#4ea1ff", width=4)
        draw.text((35, 850), "Image robot", fill="#4ea1ff", font=font)
        draw.text(
            (735, 25),
            f"Top {min(top_k, len(candidates))} dans la fenetre locale [{start}, {stop}]",
            fill="white",
            font=font,
        )

        for rank, reference_index in enumerate(ranked_indices, start=1):
            reference = streetview.records[int(reference_index)]
            y = 70 + (rank - 1) * 160
            sheet.paste(_thumbnail(reference.path, (260, 135)), (735, y))
            color = "#55d187" if rank == 1 else "#808994"
            draw.rectangle((734, y - 1, 996, y + 136), outline=color, width=4 if rank == 1 else 2)
            draw.text((1025, y + 20), f"Top {rank}: {reference.id}", fill=color, font=font)
            draw.text((1025, y + 50), f"Fichier: {reference.path.name}", fill="white", font=font)
            draw.text((1025, y + 80), f"Index: {int(reference_index)}", fill="white", font=font)
            draw.text((1025, y + 110), f"Score: {float(scores[query_index, reference_index]):.4f}", fill="white", font=font)

        state = int(trajectory[query_index])
        selected = streetview.records[state]
        gps_text = ""
        if selected.has_position:
            gps_text = f" | GPS={selected.latitude:.6f},{selected.longitude:.6f}"
        route_text = f" | route={selected.route_index}" if selected.route_index is not None else ""
        pano_text = f" | pano={selected.pano_id}" if selected.pano_id else ""
        draw.text(
            (735, 875),
            f"Trajectoire optimisee: {selected.id} ({selected.path.name}) | score={float(scores[query_index, state]):.4f}"
            f"{route_text}{gps_text}{pano_text}",
            fill="#ffcf5a",
            font=font,
        )
        sheet.save(target / f"{query_index:03d}_{query.id}.png", optimize=True)
