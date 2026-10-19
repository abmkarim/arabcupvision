"""Distance parcourue, vitesses et temps de présence par joueur."""

from __future__ import annotations

import numpy as np
import pandas as pd

MS_TO_KMH = 3.6


def track_kinematics(
    t: np.ndarray,
    xy: np.ndarray,
    max_speed_kmh: float,
    window: int = 5,
) -> dict[str, np.ndarray | float]:
    """Cinématique d'une trajectoire sur le terrain.

    Args:
        t: Instants (s), croissants.
        xy: Positions (N, 2) en mètres (NaN autorisés pour les positions inconnues).
        max_speed_kmh: Seuil anti-aberrations : un pas plus rapide est ignoré
            (ni distance, ni vitesse).
        window: Fenêtre de lissage (pas) de la vitesse instantanée.

    Returns:
        ``speed_kmh`` (N,) vitesse lissée (NaN si inconnue), ``distance_m``, ``max_speed_kmh``.
    """
    t = np.asarray(t, dtype=float)
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    n = len(t)
    speed = np.full(n, np.nan)
    if n < 2:
        return {"speed_kmh": speed, "distance_m": 0.0, "max_speed_kmh": 0.0}
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        step_speed = np.where(dt > 0, step / dt * MS_TO_KMH, np.nan)
    valid = np.isfinite(step_speed) & (step_speed <= max_speed_kmh)
    distance = float(step[valid].sum())
    raw = np.where(valid, step_speed, np.nan)
    speed[1:] = (
        pd.Series(raw).rolling(max(1, window), center=True, min_periods=1).mean().to_numpy()
    )
    finite = speed[np.isfinite(speed)]
    max_speed = float(finite.max()) if len(finite) else 0.0
    return {"speed_kmh": speed, "distance_m": distance, "max_speed_kmh": max_speed}


def player_statistics(
    tracks: pd.DataFrame,
    max_speed_kmh: float,
    window: int = 5,
    frame_interval_s: float | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Statistiques par piste à partir d'un tableau de positions terrain.

    Args:
        tracks: Colonnes ``track_id``, ``time_s``, ``pitch_x``, ``pitch_y`` (+ ``team``,
            ``role`` facultatifs).
        frame_interval_s: Durée représentée par une observation (temps de présence). Par
            défaut, l'écart médian entre instants consécutifs.

    Returns:
        Le tableau des statistiques (une ligne par piste) et la vitesse lissée de chaque
        observation (alignée sur l'index de ``tracks``).
    """
    speeds = pd.Series(np.nan, index=tracks.index, name="speed_kmh")
    rows = []
    if frame_interval_s is None:
        times = np.sort(tracks["time_s"].unique()) if len(tracks) else np.array([])
        frame_interval_s = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    for tid, g in tracks.sort_values("time_s").groupby("track_id"):
        k = track_kinematics(g["time_s"].to_numpy(), g[["pitch_x", "pitch_y"]].to_numpy(),
                             max_speed_kmh, window)
        speeds.loc[g.index] = k["speed_kmh"]
        finite = k["speed_kmh"][np.isfinite(k["speed_kmh"])]
        row = {
            "track_id": int(tid),
            "team": int(g["team"].mode().iat[0]) if "team" in g else -1,
            "role": str(g["role"].mode().iat[0]) if "role" in g else "player",
            "frames": len(g),
            "presence_s": round(len(g) * frame_interval_s, 2),
            "distance_m": round(float(k["distance_m"]), 2),
            "avg_speed_kmh": round(float(finite.mean()), 2) if len(finite) else 0.0,
            "max_speed_kmh": round(float(k["max_speed_kmh"]), 2),
            "first_seen_s": round(float(g["time_s"].min()), 2),
            "last_seen_s": round(float(g["time_s"].max()), 2),
        }
        rows.append(row)
    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats = stats.sort_values(["team", "track_id"]).reset_index(drop=True)
    return stats, speeds
