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
        dist = mm.distances.iou_matrix(g[["x", "y", "w", "h"]].to_numpy(dtype=float),
                                       h[["x", "y", "w", "h"]].to_numpy(dtype=float),
                                       max_iou=1 - iou_threshold)
        acc.update(g["id"].tolist(), h["id"].tolist(), dist, frameid=int(f))
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["idf1", "mota", "num_switches", "num_frames"],
                         name="all")
    row = summary.loc["all"]
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in row.items()}
