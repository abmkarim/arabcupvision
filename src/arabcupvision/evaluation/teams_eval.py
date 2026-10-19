"""Précision de l'assignation d'équipe à partir d'un fichier d'étiquettes."""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

import pandas as pd


def load_team_labels(path: str | Path) -> pd.DataFrame:
    """Charge un CSV ``track_id,team`` (équipe : 0/1 ou n'importe quelle étiquette textuelle)."""
    df = pd.read_csv(path)
    missing = {"track_id", "team"} - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path} : {sorted(missing)}")
    return df[["track_id", "team"]].astype({"track_id": int, "team": str})


def team_accuracy(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    """Précision par piste, invariante à la numérotation des équipes (meilleure permutation).

    Args:
        labels: Colonnes ``track_id``, ``team`` (vérité).
        predictions: Colonnes ``track_id``, ``team`` (prédiction 0/1 ; -1 = inconnue).
    """
    pred = predictions.groupby("track_id")["team"].agg(lambda s: s.mode().iat[0])
    merged = labels.merge(pred.rename("pred").reset_index(), on="track_id", how="left")
    merged["pred"] = merged["pred"].fillna(-1).astype(int)
    truth = sorted(merged["team"].unique())
    best = 0.0
    for perm in permutations(range(max(2, len(truth))), len(truth)):
        mapping = dict(zip(truth, perm, strict=True))
        acc = float((merged["team"].map(mapping) == merged["pred"]).mean())
        best = max(best, acc)
    return {"accuracy": best, "tracks_labelled": float(len(merged)),
            "tracks_unpredicted": float((merged["pred"] < 0).sum())}
