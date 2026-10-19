"""Génération d'une courte vidéo synthétique de match (tests et démonstration uniquement).

La vidéo représente un terrain en perspective avec deux équipes, deux gardiens, un arbitre
et un ballon. Elle est accompagnée d'un calibrage (JSON) et d'une vérité terrain au format
MOTChallenge, ce qui permet d'exercer tout le pipeline sans image de match réelle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from arabcupvision.pitch.geometry import PitchDimensions, draw_pitch
from arabcupvision.pitch.homography import transform_points

KIT_COLORS_BGR = {
    "team_0": ((40, 40, 200), (255, 255, 255)),  # maillot rouge, short blanc
    "team_1": ((245, 245, 245), (40, 40, 40)),  # maillot blanc, short noir
    "gk_0": ((40, 200, 230), (40, 40, 40)),
    "gk_1": ((200, 120, 30), (40, 40, 40)),
    "referee": ((25, 25, 25), (25, 25, 25)),
}


@dataclass
class SyntheticAgent:
    """Personnage animé : ancre sur le terrain, amplitude et phase de déplacement."""

    agent_id: int
    kind: str
    anchor: np.ndarray
    amplitude: np.ndarray
    freq: np.ndarray
    phase: np.ndarray

    def position(self, t: float) -> np.ndarray:
        """Position (m) à l'instant ``t`` (s)."""
        return self.anchor + self.amplitude * np.sin(2 * np.pi * self.freq * t + self.phase)


def _camera_homography(size: tuple[int, int], dims: PitchDimensions) -> np.ndarray:
    """Homographie terrain (m) → image (px) d'une caméra latérale en hauteur."""
    w, h = size
    pitch = np.array([[30, 0], [75, 0], [75, dims.width], [30, dims.width]], dtype=np.float32)
    image = np.array([[0.12 * w, 0.14 * h], [0.88 * w, 0.14 * h], [1.18 * w, 1.0 * h],
                      [-0.18 * w, 1.0 * h]], dtype=np.float32)
    return cv2.getPerspectiveTransform(pitch, image)


def _make_agents(dims: PitchDimensions, rng: np.random.Generator) -> list[SyntheticAgent]:
    agents: list[SyntheticAgent] = []
    aid = 1
    for team, x_lo, x_hi in (("team_0", 34, 52), ("team_1", 53, 71)):
        for _ in range(8):
            anchor = np.array([rng.uniform(x_lo, x_hi), rng.uniform(8, dims.width - 8)])
            agents.append(SyntheticAgent(aid, team, anchor, rng.uniform(2, 6, 2),
                                         rng.uniform(0.05, 0.15, 2), rng.uniform(0, 6.28, 2)))
            aid += 1
    for gk, x in (("gk_0", 32.0), ("gk_1", 73.0)):
        agents.append(SyntheticAgent(aid, gk, np.array([x, dims.width / 2]),
                                     np.array([0.8, 4.0]), np.array([0.05, 0.08]),
                                     rng.uniform(0, 6.28, 2)))
        aid += 1
    agents.append(SyntheticAgent(aid, "referee", np.array([dims.length / 2, dims.width / 2]),
                                 np.array([8.0, 10.0]), np.array([0.06, 0.05]),
                                 rng.uniform(0, 6.28, 2)))
    return agents


def _draw_person(img: np.ndarray, foot: np.ndarray, height: float, kind: str,
                 t: float) -> tuple[int, int, int, int]:
    """Dessine un personnage stylisé ; renvoie sa boîte englobante (x, y, w, h)."""
    shirt, shorts = KIT_COLORS_BGR[kind]
    skin, socks = (120, 160, 205), shorts
    x, y = float(foot[0]), float(foot[1])
    hgt = max(height, 12.0)
    wid = 0.36 * hgt
    swing = np.sin(t * 9.0) * 0.12 * hgt
    hip_y, shoulder_y, head_r = y - 0.48 * hgt, y - 0.80 * hgt, 0.08 * hgt
    lw = max(2, int(0.07 * hgt))
    for dx in (-swing, swing):
        cv2.line(img, (int(x), int(hip_y)), (int(x + dx), int(y)), socks, lw, cv2.LINE_AA)
    cv2.rectangle(img, (int(x - wid * 0.42), int(hip_y - 0.08 * hgt)),
                  (int(x + wid * 0.42), int(hip_y + 0.08 * hgt)), shorts, -1, cv2.LINE_AA)
    torso = np.array([[x - wid / 2, shoulder_y], [x + wid / 2, shoulder_y],
                      [x + wid * 0.42, hip_y - 0.06 * hgt], [x - wid * 0.42, hip_y - 0.06 * hgt]],
                     dtype=np.int32)
    cv2.fillPoly(img, [torso], shirt, cv2.LINE_AA)
    cv2.polylines(img, [torso], True, (30, 30, 30), 1, cv2.LINE_AA)
    for side in (-1, 1):
        cv2.line(img, (int(x + side * wid / 2), int(shoulder_y + 2)),
                 (int(x + side * wid * 0.7 - side * swing * 0.5), int(hip_y)), shirt, lw,
                 cv2.LINE_AA)
    cv2.circle(img, (int(x), int(shoulder_y - head_r * 1.2)), int(head_r), skin, -1, cv2.LINE_AA)
    top = shoulder_y - head_r * 2.2
    return int(x - wid * 0.75), int(top), int(wid * 1.5), int(y - top)


def generate_synthetic_match(
    output: str | Path,
    seconds: float = 8.0,
    fps: float = 25.0,
    size: tuple[int, int] = (1280, 720),
    seed: int = 42,
) -> dict[str, Path]:
    """Génère la vidéo, son calibrage et sa vérité terrain.

    Returns:
        Les chemins ``video``, ``calibration`` (JSON) et ``gt`` (MOTChallenge).
    """
    rng = np.random.default_rng(seed)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dims = PitchDimensions()
    H = _camera_homography(size, dims)
    ppm_top = 12
    top_view, top_scale = draw_pitch(dims, width_px=int(dims.length * ppm_top) + 20, margin_px=10,
                                     thickness=3)
    stripes = (np.arange(top_view.shape[1]) // int(5.25 * top_scale)) % 2 == 0
    grass = np.all(top_view == top_view[1, 1], axis=2)
    top_view[grass & stripes[None, :]] = (52, 118, 44)
    # Composition : pixels de la vue de dessus → mètres → image.
    to_m = np.array([[1 / top_scale, 0, -10 / top_scale], [0, 1 / top_scale, -10 / top_scale],
                     [0, 0, 1]])
    background = cv2.warpPerspective(top_view, H @ to_m, size, borderValue=(35, 70, 30))
    sky = np.zeros_like(background)
    sky[:] = (70, 60, 55)
    mask = cv2.warpPerspective(np.ones(top_view.shape[:2], np.uint8), H @ to_m, size)
    background = np.where(mask[..., None] > 0, background, sky)

    agents = _make_agents(dims, rng)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    gt_lines: list[str] = []
    n_frames = int(seconds * fps)
    for f in range(n_frames):
        t = f / fps
        frame = background.copy()
        pos = np.stack([a.position(t) for a in agents])
        feet = transform_points(H, pos)
        side = transform_points(H, pos + np.array([1.0, 0.0]))
        order = np.argsort(feet[:, 1])
        for i in order:
            a = agents[i]
            height = float(np.clip(np.linalg.norm(side[i] - feet[i]) * 1.8, 18, 200))
            x, y, w, h = _draw_person(frame, feet[i], height, a.kind, t + a.phase[0])
            gt_lines.append(f"{f + 1},{a.agent_id},{x},{y},{w},{h},1,-1,-1,-1")
        carrier = agents[(f // int(2 * fps)) % 16].position(t)
        ball_m = carrier + np.array([0.8, 0.4])
        bx, by = transform_points(H, ball_m[None])[0]
        r = max(3, int(0.004 * size[0]))
        cv2.circle(frame, (int(bx), int(by)), r, (250, 250, 250), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(bx), int(by)), r, (60, 60, 60), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()

    H_inv = np.linalg.inv(H)
    corners = np.array([[0.25 * size[0], 0.3 * size[1]], [0.75 * size[0], 0.3 * size[1]],
                        [0.9 * size[0], 0.9 * size[1]], [0.1 * size[0], 0.9 * size[1]]])
    calib = {
        "image_points": corners.round(2).tolist(),
        "pitch_points": transform_points(H_inv, corners).round(3).tolist(),
    }
    calib_path = out.with_suffix(".calibration.json")
    calib_path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
    gt_path = out.with_suffix(".gt.txt")
    gt_path.write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    return {"video": out, "calibration": calib_path, "gt": gt_path}
