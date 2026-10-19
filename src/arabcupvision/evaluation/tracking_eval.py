"""Évaluation du suivi (IDF1, MOTA, changements d'identité) avec motmetrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MOT_COLUMNS = ["frame", "id", "x", "y", "w", "h", "conf", "cx", "cy", "cz"]


def load_mot(path: str | Path) -> pd.DataFrame:
    """Charge un fichier MOTChallenge (``frame,id,x,y,w,h,conf,...``, images numérotées dès 1)."""
    df = pd.read_csv(path, header=None)
    df = df.iloc[:, : len(MOT_COLUMNS)]
    df.columns = MOT_COLUMNS[: df.shape[1]]
    return df[["frame", "id", "x", "y", "w", "h"]]


def tracks_to_mot(tracks: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """Convertit le ``tracks.csv`` du pipeline au format MOT (coordonnées natives)."""
    df = pd.DataFrame({
        "frame": tracks["frame"].astype(int) + 1,
        "id": tracks["track_id"].astype(int),
        "x": tracks["x1"] / scale,
        "y": tracks["y1"] / scale,
        "w": (tracks["x2"] - tracks["x1"]) / scale,
        "h": (tracks["y2"] - tracks["y1"]) / scale,
    })
    return df


def iou_distance(a: np.ndarray, b: np.ndarray, max_distance: float) -> np.ndarray:
    """Matrice ``1 - IoU`` entre boîtes ``xywh`` ; NaN au-delà de ``max_distance`` (non appariable).

    Équivalent de ``motmetrics.distances.iou_matrix``, compatible NumPy 2.
    """
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.empty((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]),
                 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]),
                 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    dist = 1.0 - np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    dist[dist > max_distance] = np.nan
    return dist


def evaluate_tracking(gt: pd.DataFrame, hyp: pd.DataFrame, iou_threshold: float = 0.5,
                      frames: list[int] | None = None) -> dict[str, float]:
    """Calcule IDF1, MOTA et le nombre de changements d'identité.

    Args:
        gt: Vérité terrain au format MOT.
        hyp: Hypothèses au format MOT.
        frames: Images évaluées (par défaut, celles présentes dans les hypothèses).
    """
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=False)
    if frames is None:
        frames = sorted(set(hyp["frame"]))
    gt_by = dict(tuple(gt.groupby("frame")))
    hyp_by = dict(tuple(hyp.groupby("frame")))
    empty = pd.DataFrame(columns=["id", "x", "y", "w", "h"])
    for f in frames:
        g = gt_by.get(f, empty)
        h = hyp_by.get(f, empty)
        dist = iou_distance(g[["x", "y", "w", "h"]].to_numpy(dtype=float),
                            h[["x", "y", "w", "h"]].to_numpy(dtype=float), 1 - iou_threshold)
        acc.update(g["id"].tolist(), h["id"].tolist(), dist, frameid=int(f))
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["idf1", "mota", "num_switches", "num_frames"],
                         name="all")
    row = summary.loc["all"]
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in row.items()}
