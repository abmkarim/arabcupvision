"""Heatmaps de présence sur le terrain."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from arabcupvision.pitch.geometry import PitchDimensions, draw_pitch  # noqa: E402


def heatmap_grid(
    xy: np.ndarray,
    length: float,
    width: float,
    bins: tuple[int, int] = (42, 27),
    sigma: float = 1.5,
) -> np.ndarray:
    """Histogramme 2D lissé et normalisé (max = 1) des positions sur le terrain.

    Returns:
        Une grille de forme ``(bins[1], bins[0])`` (lignes = largeur, colonnes = longueur).
    """
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] <= length) & (pts[:, 1] >= 0) & (pts[:, 1] <= width)]
    grid, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=(bins[1], bins[0]),
                                range=((0, width), (0, length)))
    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma)
    peak = grid.max()
    return grid / peak if peak > 0 else grid


def render_heatmap(
    grid: np.ndarray,
    dims: PitchDimensions,
    title: str = "",
    path: str | Path | None = None,
    cmap: str = "inferno",
) -> plt.Figure:
    """Dessine la heatmap au-dessus d'un terrain vu de dessus (et l'enregistre si ``path``)."""
    margin = 10
    pitch_img, scale = draw_pitch(dims, width_px=840, margin_px=margin)
    h, w = pitch_img.shape[:2]
    fig, ax = plt.subplots(figsize=(8.4, h / 100), dpi=100)
    ax.imshow(cv2.cvtColor(pitch_img, cv2.COLOR_BGR2RGB))
    extent = (margin, margin + dims.length * scale, margin + dims.width * scale, margin)
    masked = np.ma.masked_less(grid, 0.02)
    ax.imshow(masked, extent=extent, cmap=cmap, alpha=0.7, interpolation="bilinear",
              vmin=0, vmax=1)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100)
        plt.close(fig)
    return fig
