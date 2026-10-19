"""Détection des points clés du terrain (YOLOv8-pose) et conversion en homographie."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from arabcupvision.config import PitchConfig
from arabcupvision.pitch.homography import estimate_homography, transform_points

log = logging.getLogger(__name__)


class PitchKeypointDetector:
    """Prédit les points clés du terrain et en déduit l'homographie image → terrain."""

    def __init__(self, model_path: Path, cfg: PitchConfig, device: str = "cpu") -> None:
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = device
        self.model = YOLO(str(model_path))
        self.reference = np.asarray(cfg.keypoints_m, dtype=np.float64).reshape(-1, 2)
        if len(self.reference) == 0:
            raise ValueError("pitch.keypoints_m est vide : correspondances inconnues.")

    def predict(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Renvoie les points image et terrain des points clés suffisamment confiants."""
        result = self.model.predict(frame, device=self.device, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return np.empty((0, 2)), np.empty((0, 2))
        xy = result.keypoints.xy[0].cpu().numpy()
        conf = result.keypoints.conf
        conf = conf[0].cpu().numpy() if conf is not None else np.ones(len(xy))
        n = min(len(xy), len(self.reference))
        mask = (conf[:n] >= self.cfg.keypoint_confidence) & (xy[:n, 0] > 1) & (xy[:n, 1] > 1)
        return xy[:n][mask], self.reference[:n][mask]

    def homography(self, frame: np.ndarray) -> np.ndarray | None:
        """Calcule l'homographie de l'image, ou ``None`` si la confiance est insuffisante."""
        img_pts, pitch_pts = self.predict(frame)
        if len(img_pts) < self.cfg.min_keypoints:
            return None
        H = estimate_homography(img_pts, pitch_pts, self.cfg.ransac_threshold_m)
        if H is None:
            return None
        err = np.linalg.norm(transform_points(H, img_pts) - pitch_pts, axis=1)
        if np.median(err) > 3 * self.cfg.ransac_threshold_m:
            return None
        return H
