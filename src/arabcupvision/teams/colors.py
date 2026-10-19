"""Extraction de la couleur dominante du maillot d'un joueur."""

from __future__ import annotations

import cv2
import numpy as np

from arabcupvision.config import TeamsConfig


def jersey_crop(frame: np.ndarray, box: np.ndarray, cfg: TeamsConfig) -> np.ndarray:
    """Découpe la zone du haut du maillot (partie supérieure centrale de la boîte)."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    top = int(round(y1 + cfg.crop_top * h))
    bottom = int(round(y1 + cfg.crop_bottom * h))
    left = int(round(x1 + cfg.crop_side * w))
    right = int(round(x2 - cfg.crop_side * w))
    top, bottom = max(0, top), min(h_img, max(bottom, top + 1))
    left, right = max(0, left), min(w_img, max(right, left + 1))
    return frame[top:bottom, left:right]


def grass_mask(crop_bgr: np.ndarray, cfg: TeamsConfig) -> np.ndarray:
    """Masque booléen des pixels de pelouse (teinte verte suffisamment saturée)."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lo, hi = cfg.green_hue_range
    return (hsv[..., 0] >= lo) & (hsv[..., 0] <= hi) & (hsv[..., 1] >= cfg.green_min_saturation)


def dominant_color_lab(crop_bgr: np.ndarray, cfg: TeamsConfig) -> np.ndarray | None:
    """Couleur dominante (médiane Lab) hors pelouse, ou ``None`` si trop peu de pixels."""
    if crop_bgr.size == 0:
        return None
    keep = ~grass_mask(crop_bgr, cfg)
    if int(keep.sum()) < cfg.min_pixels:
        return None
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)[keep.reshape(-1)]
    return np.median(lab.astype(np.float32), axis=0)


def jersey_color(frame: np.ndarray, box: np.ndarray, cfg: TeamsConfig) -> np.ndarray | None:
    """Couleur Lab du maillot d'une détection."""
    return dominant_color_lab(jersey_crop(frame, box, cfg), cfg)


def lab_to_bgr(lab: np.ndarray) -> tuple[int, int, int]:
    """Convertit une couleur Lab OpenCV (0-255) en tuple BGR."""
    pixel = np.clip(np.asarray(lab, dtype=np.float32), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    b, g, r = cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)[0, 0]
    return int(b), int(g), int(r)
