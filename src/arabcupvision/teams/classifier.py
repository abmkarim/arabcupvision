"""Regroupement des joueurs en deux équipes : K-Means sur les couleurs puis vote par piste."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

UNKNOWN_TEAM = -1


class TeamClassifier:
    """K-Means à deux classes sur les couleurs Lab des maillots."""

    def __init__(self, seed: int = 42, outlier_factor: float = 2.5,
                 min_outlier_distance: float = 25.0) -> None:
        self.seed = seed
        self.outlier_factor = outlier_factor
        self.min_outlier_distance = min_outlier_distance
        self.kmeans: KMeans | None = None
        self.spread: np.ndarray = np.zeros(2)
        self.robust_centers: np.ndarray = np.zeros((2, 3))

    @property
    def is_fitted(self) -> bool:
        """Indique si le modèle a été ajusté."""
        return self.kmeans is not None

    @property
    def centers(self) -> np.ndarray:
        """Centres Lab des deux équipes."""
        if self.kmeans is None:
            raise RuntimeError("TeamClassifier non ajusté.")
        return self.robust_centers

    def fit(self, colors: np.ndarray) -> TeamClassifier:
        """Ajuste le K-Means, puis le ré-ajuste sans les couleurs atypiques (arbitre, gardiens).

        Au moins deux couleurs sont nécessaires.
        """
        X = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if len(X) < 2:
            raise ValueError("Au moins deux couleurs sont nécessaires pour séparer deux équipes.")
        self._fit_once(X)
        inliers = X[~self.is_outlier(X)]
        if len(inliers) < len(X) and len(np.unique(inliers, axis=0)) >= 2:
            self._fit_once(inliers)
        return self

    def _fit_once(self, X: np.ndarray) -> None:
        # Centres robustes : médiane de chaque cluster, peu sensible aux couleurs minoritaires.
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=self.seed).fit(X)
        lbl = self.kmeans.labels_
        self.robust_centers = np.stack([
            np.median(X[lbl == k], axis=0) if np.any(lbl == k) else self.kmeans.cluster_centers_[k]
            for k in range(2)
        ])
        dist = self._distances(X)
        self.spread = np.array([
            float(np.median(dist[lbl == k, k])) if np.any(lbl == k) else 0.0 for k in range(2)
        ])

    def _distances(self, X: np.ndarray) -> np.ndarray:
        return np.linalg.norm(X[:, None, :] - self.robust_centers[None, :, :], axis=2)

    def predict(self, colors: np.ndarray) -> np.ndarray:
        """Équipe (0 ou 1) la plus proche de chaque couleur."""
        X = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if len(X) == 0:
            return np.empty(0, dtype=int)
        if self.kmeans is None:
            return np.full(len(X), UNKNOWN_TEAM)
        return np.argmin(self._distances(X), axis=1).astype(int)

    def is_outlier(self, colors: np.ndarray) -> np.ndarray:
        """Couleurs éloignées de leur équipe (arbitre, gardien) : seuil propre à chaque cluster."""
        X = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if self.kmeans is None or len(X) == 0:
            return np.zeros(len(X), dtype=bool)
        dist = self._distances(X)
        lbl = np.argmin(dist, axis=1)
        threshold = np.maximum(self.outlier_factor * self.spread[lbl], self.min_outlier_distance)
        return dist[np.arange(len(X)), lbl] > threshold


def majority_vote(track_ids: Iterable[int], labels: Iterable[int]) -> dict[int, int]:
    """Étiquette majoritaire par piste (les étiquettes inconnues sont ignorées).

    En cas d'égalité, l'étiquette la plus petite l'emporte (résultat déterministe).
    """
    df = pd.DataFrame({"track_id": list(track_ids), "label": list(labels)})
    df = df[df["label"] != UNKNOWN_TEAM]
    if df.empty:
        return {}
    counts = df.groupby(["track_id", "label"]).size().reset_index(name="n")
    counts = counts.sort_values(["track_id", "n", "label"], ascending=[True, False, True])
    best = counts.drop_duplicates("track_id")
    return {int(t): int(lbl) for t, lbl in zip(best["track_id"], best["label"], strict=True)}


def assign_by_centroid(
    positions: dict[int, np.ndarray], team_centroids: dict[int, np.ndarray]
) -> dict[int, int]:
    """Rattache chaque piste (ex. gardien) à l'équipe dont le centre de gravité est le plus proche.

    Args:
        positions: Position moyenne (x, y) de chaque piste à rattacher.
        team_centroids: Centre de gravité (x, y) de chaque équipe.
    """
    if not team_centroids:
        return {tid: UNKNOWN_TEAM for tid in positions}
    teams = sorted(team_centroids)
    C = np.stack([np.asarray(team_centroids[t], dtype=float) for t in teams])
    out: dict[int, int] = {}
    for tid, pos in positions.items():
        d = np.linalg.norm(C - np.asarray(pos, dtype=float), axis=1)
        out[tid] = teams[int(np.argmin(d))]
    return out
