from __future__ import annotations

from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from dev_matching.data import ImageCollection
from dev_matching.filtering import RosaFilterResult

from .matches import _thumbnail


def write_filter_overview(
    path: str | Path,
    streetview: ImageCollection,
    result: RosaFilterResult,
    columns: int = 4,
) -> None:
    """Render every reference with its ROSA keep/reject decision."""
    cell_width, cell_height = 340, 225
    rows = ceil(len(streetview) / columns)
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "#111418")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for decision in result.decisions:
        index = decision.original_index
        record = streetview.records[index]
        row, column = divmod(index, columns)
        x, y = column * cell_width, row * cell_height
        canvas.paste(_thumbnail(record.path, (320, 160)), (x + 10, y + 10))
        color = "#55d187" if decision.selected else "#e16b6b"
        draw.rectangle((x + 9, y + 9, x + 331, y + 171), outline=color, width=4)
        label = f"{index:02d} {record.id} | {'GARDEE' if decision.selected else 'REJETEE'}"
        draw.text((x + 10, y + 180), label, fill=color, font=font)
        draw.text((x + 10, y + 198), decision.reason, fill="white", font=font)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, optimize=True)
