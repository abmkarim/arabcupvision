"""Possession par équipe : joueur le plus proche du ballon, sous un seuil de distance."""

from __future__ import annotations

import numpy as np
import pandas as pd


def possession_per_frame(
    players: pd.DataFrame,
    ball: pd.DataFrame,
    max_distance: float,
    x: str = "pitch_x",
    y: str = "pitch_y",
) -> pd.DataFrame:
    """Équipe en possession à chaque image.

    Args:
        players: Colonnes ``frame``, ``track_id``, ``team``, ``x``, ``y`` (équipes 0/1).
        ball: Colonnes ``frame``, ``x``, ``y`` (une ligne par image).
        max_distance: Distance maximale joueur-ballon pour attribuer la possession.

    Returns:
        Colonnes ``frame``, ``team`` (-1 si personne), ``track_id`` (-1), ``distance``.
    """
    rows = []
    teams = players[players["team"] >= 0]
    by_frame = dict(tuple(teams.groupby("frame")))
    for _, b in ball.iterrows():
        f = int(b["frame"])
        team, tid, dist = -1, -1, np.nan
        g = by_frame.get(f)
        if g is not None and np.isfinite(b[x]) and np.isfinite(b[y]):
            d = np.hypot(g[x].to_numpy() - b[x], g[y].to_numpy() - b[y])
            if np.isfinite(d).any():
                i = int(np.nanargmin(d))
                dist = float(d[i])
                if dist <= max_distance:
                    team, tid = int(g["team"].iat[i]), int(g["track_id"].iat[i])
        rows.append({"frame": f, "team": team, "track_id": tid, "distance": dist})
    return pd.DataFrame(rows, columns=["frame", "team", "track_id", "distance"])


def possession_share(per_frame: pd.DataFrame) -> dict[str, float]:
    """Part de possession (%) de chaque équipe parmi les images attribuées."""
    owned = per_frame[per_frame["team"] >= 0]
    if owned.empty:
        return {"team_0": 0.0, "team_1": 0.0}
    share = owned["team"].value_counts(normalize=True) * 100
    return {f"team_{t}": round(float(share.get(t, 0.0)), 1) for t in (0, 1)}
