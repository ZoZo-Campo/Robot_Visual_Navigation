from __future__ import annotations

from pathlib import Path

import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; run scripts/setup_mac.sh") from exc
    return plt


def plot_heatmap(similarity: np.ndarray, output: str | Path, title: str = "Similarity matrix") -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(12, 7))
    image = axis.imshow(similarity, aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set(title=title, xlabel="Street View index", ylabel="Robot frame index")
    figure.colorbar(image, ax=axis, label="Similarity")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_timeline(states: np.ndarray, output: str | Path, title: str = "Decoded trajectory") -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(np.arange(len(states)), states, marker="o", markersize=3)
    axis.set(title=title, xlabel="Robot frame index", ylabel="Street View index")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
