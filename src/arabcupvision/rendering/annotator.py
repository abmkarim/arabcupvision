"""Annotation des images : ellipses au sol, numéros de piste, ballon et mini-carte."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from arabcupvision.config import RenderConfig
from arabcupvision.pitch.geometry import PitchDimensions, draw_pitch, pitch_to_canvas


@dataclass
class FrameItem:
    """Objet à dessiner sur une image."""

    box: np.ndarray
    track_id: int
    color: tuple[int, int, int]
    pitch_xy: tuple[float, float] | None = None
    label: str | None = None


def _text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg
    return (0, 0, 0) if 0.114 * b + 0.587 * g + 0.299 * r > 140 else (255, 255, 255)


def draw_player(frame: np.ndarray, item: FrameItem) -> None:
    """Ellipse partielle sous les pieds et étiquette avec le numéro de piste."""
    x1, _, x2, y2 = (float(v) for v in item.box)
    cx, cy = int(round((x1 + x2) / 2)), int(round(y2))
    w = max(8, int(round(x2 - x1)))
    axes = (w // 2 + 2, max(4, int(0.3 * w)))
    cv2.ellipse(frame, (cx, cy), axes, 0, -45, 235, item.color, 2, cv2.LINE_AA)
    text = item.label if item.label is not None else str(item.track_id)
    if not text:
        return
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (tw, tht), _ = cv2.getTextSize(text, font, fs, th)
    top = cy + axes[1] - 2
    p1 = (cx - tw // 2 - 4, top)
    p2 = (cx + tw // 2 + 4, top + tht + 6)
    cv2.rectangle(frame, p1, p2, item.color, -1, cv2.LINE_AA)
    cv2.putText(frame, text, (cx - tw // 2, top + tht + 2), font, fs, _text_color(item.color),
                th, cv2.LINE_AA)


def draw_ball(frame: np.ndarray, xy: tuple[float, float], color: tuple[int, int, int],
              interpolated: bool = False) -> None:
    """Triangle pointant vers le ballon (contour seul si la position est interpolée)."""
    x, y = int(round(xy[0])), int(round(xy[1]))
    pts = np.array([[x, y - 8], [x - 8, y - 22], [x + 8, y - 22]], dtype=np.int32)
    if interpolated:
        cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
    else:
        cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)
        cv2.polylines(frame, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)


class MiniMap:
    """Mini-carte tactique : terrain vu de dessus avec la position des joueurs."""

    def __init__(self, dims: PitchDimensions, cfg: RenderConfig) -> None:
        self.dims = dims
        self.cfg = cfg
        self.margin = 8
        self.base, self.scale = draw_pitch(dims, width_px=cfg.minimap_width,
                                           margin_px=self.margin, thickness=1)

    def render(self, items: list[FrameItem], ball_xy: tuple[float, float] | None) -> np.ndarray:
        """Dessine la mini-carte de l'image courante."""
        img = self.base.copy()
        for it in items:
            if it.pitch_xy is None or not np.all(np.isfinite(it.pitch_xy)):
                continue
            px, py = pitch_to_canvas(np.array(it.pitch_xy), self.scale, self.margin)[0]
            cv2.circle(img, (int(px), int(py)), 5, it.color, -1, cv2.LINE_AA)
            cv2.circle(img, (int(px), int(py)), 5, (0, 0, 0), 1, cv2.LINE_AA)
        if ball_xy is not None and np.all(np.isfinite(ball_xy)):
            bx, by = pitch_to_canvas(np.array(ball_xy), self.scale, self.margin)[0]
            cv2.circle(img, (int(bx), int(by)), 3, self.cfg.ball_color_bgr, -1, cv2.LINE_AA)
            cv2.circle(img, (int(bx), int(by)), 3, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    def overlay(self, frame: np.ndarray, minimap: np.ndarray) -> None:
        """Incruste la mini-carte en bas au centre de l'image."""
        fh, fw = frame.shape[:2]
        mh, mw = minimap.shape[:2]
        if mw > fw or mh > fh:
            ratio = min(fw / mw, fh / mh) * 0.9
            minimap = cv2.resize(minimap, (int(mw * ratio), int(mh * ratio)))
            mh, mw = minimap.shape[:2]
        x0, y0 = (fw - mw) // 2, fh - mh - 10
        roi = frame[y0 : y0 + mh, x0 : x0 + mw]
        a = self.cfg.minimap_alpha
        frame[y0 : y0 + mh, x0 : x0 + mw] = cv2.addWeighted(minimap, a, roi, 1 - a, 0)


def draw_header(frame: np.ndarray, text: str) -> None:
    """Bandeau d'information en haut à gauche."""
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.55
    (tw, th), _ = cv2.getTextSize(text, font, fs, 1)
    cv2.rectangle(frame, (8, 8), (20 + tw, 20 + th), (20, 20, 20), -1)
    cv2.putText(frame, text, (14, 14 + th), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
