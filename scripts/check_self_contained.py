from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("Dev" + "_Matching_V2", "DEV" + "_MATCHING_ROOT")
REQUIRED = (
    ROOT / "vpr_backend" / "dev_matching" / "data" / "collection.py",
    ROOT / "vpr_backend" / "dev_matching" / "matchers" / "mixvpr.py",
    ROOT / "weights" / "resnet50_MixVPR_4096.ckpt",
)


def main() -> int:
    for path in REQUIRED:
        if not path.exists():
            print(f"Missing required local file: {path}")
            return 1

    for path in ROOT.rglob("*.py"):
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        if path == Path(__file__).resolve():
            continue
        ast.parse(path.read_text(encoding="utf-8"))
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN:
            if forbidden in text:
                print(f"Forbidden external backend reference in {path}: {forbidden}")
                return 1

    sys.path.insert(0, str(ROOT))
    from matching.vpr_matcher import VPRMatcher

    matcher = VPRMatcher(
        ROOT / "data" / "streetview" / "images",
        ROOT / "data" / "streetview" / "metadata.csv",
        mode="Lightweight VPR",
    )
    result = matcher.search(ROOT / "data" / "frames" / "frame_00000.jpg", top_k=1)[0]
    print(
        "Self-contained V3 OK:",
        matcher.summary["references_after_filtering"],
        result["filename"],
        f"{result['score']:.2f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
