"""Post-traitement des trajectoires : lissage par piste et interpolation du ballon."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def smooth_tracks(
    df: pd.DataFrame,
    columns: Sequence[str],
    window: int,
    group: str = "track_id",
    order: str = "frame",
) -> pd.DataFrame:
    """Lisse des colonnes par moyenne glissante centrée, piste par piste.

    Les valeurs manquantes restent manquantes ; les bords utilisent une fenêtre partielle.
    """
    out = df.sort_values([group, order]).copy()
    if window <= 1 or out.empty:
        return out
    for col in columns:
        out[col] = out.groupby(group, sort=False)[col].transform(
            lambda s: s.rolling(window, center=True, min_periods=1).mean().where(s.notna())
        )
    return out


def interpolate_ball(
    ball: pd.DataFrame,
    frames: Sequence[int],
    max_gap: int,
    columns: Sequence[str] = ("x", "y"),
) -> pd.DataFrame:
    """Interpole linéairement les positions manquantes du ballon.

    Args:
        ball: Détections du ballon (colonne ``frame`` + ``columns``), au plus une par image.
        frames: Toutes les images traitées (ordre croissant).
        max_gap: Nombre maximal d'images manquantes consécutives comblées.

    Returns:
        Un tableau indexé par image, avec une colonne booléenne ``interpolated``.
    """
    idx = pd.Index(sorted(set(int(f) for f in frames)), name="frame")
    cols = list(columns)
    if ball.empty:
        res = pd.DataFrame(np.nan, index=idx, columns=cols)
        res["interpolated"] = False
        return res.reset_index()
    observed = ball.drop_duplicates("frame").set_index("frame")[cols].reindex(idx)
    missing = observed[cols[0]].isna()
    filled = observed.interpolate(method="index", limit_area="inside")
    # Ne combler que les trous dont la longueur ne dépasse pas max_gap.
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    too_long = missing & (run_len > max_gap)
    filled.loc[too_long, cols] = np.nan
    filled["interpolated"] = missing & filled[cols[0]].notna()
    return filled.reset_index()
