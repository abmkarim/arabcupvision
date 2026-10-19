"""Géométrie d'un terrain réglementaire et dessin vu de dessus."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PITCH_GREEN_BGR = (60, 130, 50)
LINE_WHITE_BGR = (240, 240, 240)


@dataclass(frozen=True)
class PitchDimensions:
    """Dimensions (en mètres) des marquages d'un terrain de football."""

    length: float = 105.0
    width: float = 68.0
    penalty_depth: float = 16.5
    penalty_width: float = 40.32
    goal_area_depth: float = 5.5
    goal_area_width: float = 18.32
    center_circle_radius: float = 9.15
    penalty_spot: float = 11.0

    def segments(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Renvoie les segments rectilignes du marquage, en mètres (origine coin haut-gauche)."""
        L, W = self.length, self.width
        cy = W / 2
        segs: list[tuple[tuple[float, float], tuple[float, float]]] = [
            ((0, 0), (L, 0)),
            ((L, 0), (L, W)),
            ((L, W), (0, W)),
            ((0, W), (0, 0)),
            ((L / 2, 0), (L / 2, W)),
        ]
        for x0, sign in ((0.0, 1.0), (L, -1.0)):
            for depth, width in (
                (self.penalty_depth, self.penalty_width),
                (self.goal_area_depth, self.goal_area_width),
            ):
                x1 = x0 + sign * depth
                y_top, y_bot = cy - width / 2, cy + width / 2
                segs += [((x0, y_top), (x1, y_top)), ((x1, y_top), (x1, y_bot)),
                         ((x1, y_bot), (x0, y_bot))]
        return segs

    def landmarks(self) -> dict[str, tuple[float, float]]:
        """Points remarquables nommés (coins, surfaces, rond central), en mètres."""
        L, W = self.length, self.width
        cy = W / 2
        pts: dict[str, tuple[float, float]] = {
            "corner_tl": (0, 0), "corner_tr": (L, 0), "corner_br": (L, W), "corner_bl": (0, W),
            "mid_top": (L / 2, 0), "mid_bottom": (L / 2, W), "center": (L / 2, cy),
            "circle_top": (L / 2, cy - self.center_circle_radius),
            "circle_bottom": (L / 2, cy + self.center_circle_radius),
        }
        for side, x0, sign in (("left", 0.0, 1.0), ("right", L, -1.0)):
            for name, depth, width in (
                ("box", self.penalty_depth, self.penalty_width),
                ("six", self.goal_area_depth, self.goal_area_width),
            ):
                x1 = x0 + sign * depth
                pts[f"{side}_{name}_line_top"] = (x0, cy - width / 2)
                pts[f"{side}_{name}_line_bottom"] = (x0, cy + width / 2)
                pts[f"{side}_{name}_top"] = (x1, cy - width / 2)
                pts[f"{side}_{name}_bottom"] = (x1, cy + width / 2)
            pts[f"{side}_spot"] = (x0 + sign * self.penalty_spot, cy)
        return pts


def draw_pitch(
    dims: PitchDimensions,
    width_px: int = 525,
    margin_px: int = 10,
    background: tuple[int, int, int] = PITCH_GREEN_BGR,
    line_color: tuple[int, int, int] = LINE_WHITE_BGR,
    thickness: int = 2,
) -> tuple[np.ndarray, float]:
    """Dessine le terrain vu de dessus.

    Returns:
        L'image BGR et l'échelle (pixels par mètre). Un point (x, y) en mètres se trouve
        au pixel ``(margin + x * scale, margin + y * scale)``.
    """
    scale = (width_px - 2 * margin_px) / dims.length
    height_px = int(round(dims.width * scale + 2 * margin_px))
    img = np.full((height_px, width_px, 3), background, dtype=np.uint8)

    def to_px(p: tuple[float, float]) -> tuple[int, int]:
        return int(round(margin_px + p[0] * scale)), int(round(margin_px + p[1] * scale))

    for a, b in dims.segments():
        cv2.line(img, to_px(a), to_px(b), line_color, thickness, cv2.LINE_AA)
    center = to_px((dims.length / 2, dims.width / 2))
    cv2.circle(img, center, int(round(dims.center_circle_radius * scale)), line_color,
               thickness, cv2.LINE_AA)
    cv2.circle(img, center, max(2, thickness), line_color, -1, cv2.LINE_AA)
    for side_x, sign in ((0.0, 1.0), (dims.length, -1.0)):
        spot = to_px((side_x + sign * dims.penalty_spot, dims.width / 2))
        cv2.circle(img, spot, max(2, thickness), line_color, -1, cv2.LINE_AA)
    return img, scale


def pitch_to_canvas(points_m: np.ndarray, scale: float, margin_px: int = 10) -> np.ndarray:
    """Convertit des coordonnées terrain (m) en pixels d'une image produite par ``draw_pitch``."""
    pts = np.asarray(points_m, dtype=float).reshape(-1, 2)
    return margin_px + pts * scale
