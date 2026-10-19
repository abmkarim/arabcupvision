"""Homographie image → terrain (mètres) à partir de correspondances de points."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def estimate_homography(
    image_pts: np.ndarray,
    pitch_pts: np.ndarray,
    ransac_threshold: float = 1.0,
) -> np.ndarray | None:
    """Estime la matrice 3×3 qui projette des pixels vers des mètres.

    Args:
        image_pts: Points image (N, 2) en pixels.
        pitch_pts: Points terrain correspondants (N, 2) en mètres.
        ransac_threshold: Erreur de reprojection tolérée (m) par RANSAC.

    Returns:
        La matrice, ou ``None`` si l'estimation échoue (moins de 4 points, configuration dégénérée).
    """
    src = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(pitch_pts, dtype=np.float64).reshape(-1, 2)
    if len(src) < 4 or len(src) != len(dst):
        return None
    method = cv2.RANSAC if len(src) > 4 else 0
    H, _ = cv2.findHomography(src, dst, method, ransac_threshold)
    if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
        return None
    return H


def transform_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Applique une homographie à des points (N, 2)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H)
    return out.reshape(-1, 2)


def approximate_homography(frame_w: int, frame_h: int, length: float, width: float) -> np.ndarray:
    """Homographie grossière supposant que l'image couvre l'intégralité du terrain."""
    src = np.array([[0, 0], [frame_w, 0], [frame_w, frame_h], [0, frame_h]], dtype=np.float64)
    dst = np.array([[0, 0], [length, 0], [length, width], [0, width]], dtype=np.float64)
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))


def load_calibration(path: str | Path, scale: float = 1.0) -> np.ndarray:
    """Charge un calibrage manuel (JSON) et renvoie l'homographie correspondante.

    Format attendu::

        {"image_points": [[x, y], ...], "pitch_points": [[X, Y], ...]}

    ``image_points`` est exprimé dans la résolution native de la vidéo ; ``scale`` le ramène
    à la résolution de travail.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    img = np.asarray(data["image_points"], dtype=np.float64) * scale
    pitch = np.asarray(data["pitch_points"], dtype=np.float64)
    H = estimate_homography(img, pitch, ransac_threshold=1.0)
    if H is None:
        raise ValueError(f"Calibrage invalide : {path}")
    return H


class HomographyTracker:
    """Conserve la dernière homographie valide et la met à jour par intervalle."""

    def __init__(self, recompute_interval: int = 1) -> None:
        self.recompute_interval = max(1, recompute_interval)
        self.current: np.ndarray | None = None
        self.source: str = "none"
        self._step = 0

    def should_update(self) -> bool:
        """Indique si l'image courante doit déclencher un recalcul."""
        due = self._step % self.recompute_interval == 0 or self.current is None
        self._step += 1
        return due

    def update(self, H: np.ndarray | None, source: str) -> None:
        """Remplace l'homographie courante si ``H`` est valide."""
        if H is not None:
            self.current = H
            self.source = source
