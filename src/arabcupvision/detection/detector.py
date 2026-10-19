"""Chargement du modèle de détection et inférence avec suivi multi-objets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np

from arabcupvision.config import DetectionConfig, TrackingConfig

log = logging.getLogger(__name__)


class ObjectClass(IntEnum):
    """Classes canoniques manipulées par le pipeline."""

    BALL = 0
    GOALKEEPER = 1
    PLAYER = 2
    REFEREE = 3


def canonical_class(name: str) -> ObjectClass | None:
    """Associe un nom de classe de modèle à une classe canonique (ou ``None`` si inutile)."""
    n = name.lower().replace("_", " ").strip()
    if "ball" in n:
        return ObjectClass.BALL
    if "goal" in n and "keeper" in n:
        return ObjectClass.GOALKEEPER
    if "referee" in n or n == "ref":
        return ObjectClass.REFEREE
    if n in {"player", "players", "person"}:
        return ObjectClass.PLAYER
    return None


@dataclass
class Detections:
    """Détections d'une image : boîtes ``xyxy``, confiances, classes et identifiants de piste."""

    xyxy: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))
    confidence: np.ndarray = field(default_factory=lambda: np.empty(0))
    class_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    track_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    def __len__(self) -> int:
        return len(self.xyxy)

    def subset(self, mask: np.ndarray) -> Detections:
        """Renvoie les détections sélectionnées par un masque booléen."""
        return Detections(self.xyxy[mask], self.confidence[mask], self.class_id[mask],
                          self.track_id[mask])

    @property
    def foot_points(self) -> np.ndarray:
        """Point au sol de chaque boîte (milieu du bord inférieur)."""
        if len(self) == 0:
            return np.empty((0, 2))
        return np.stack([(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, self.xyxy[:, 3]], axis=1)


class Detector:
    """Détecteur YOLOv8 avec suivi ByteTrack/BoT-SORT intégré d'ultralytics.

    Utilise le modèle fine-tuné s'il existe, sinon le modèle COCO pré-entraîné (repli) où
    seules les classes ``person`` et ``sports ball`` sont conservées.
    """

    def __init__(self, det_cfg: DetectionConfig, track_cfg: TrackingConfig, root: Path) -> None:
        from ultralytics import YOLO

        self.cfg = det_cfg
        self.track_cfg = track_cfg
        model_path = Path(det_cfg.model_path)
        if not model_path.is_absolute():
            model_path = root / model_path
        if model_path.exists():
            self.model = YOLO(str(model_path))
            self.fallback = False
        else:
            log.warning("Modèle %s absent : repli sur %s (COCO).", model_path,
                        det_cfg.fallback_model)
            fallback = root / "models" / det_cfg.fallback_model
            fallback.parent.mkdir(parents=True, exist_ok=True)
            self.model = YOLO(str(fallback))  # téléchargé dans models/ si absent
            self.fallback = True
        # Un modèle de type COCO (classe « person ») est traité comme le repli.
        self.fallback = self.fallback or "person" in {n.lower() for n in self.model.names.values()}
        self.class_map: dict[int, ObjectClass] = {}
        for idx, name in self.model.names.items():
            if self.fallback and name.lower() not in {"person", "sports ball"}:
                continue
            canon = canonical_class(name)
            if canon is not None:
                self.class_map[int(idx)] = canon
        if not self.class_map:
            raise ValueError(f"Aucune classe exploitable dans le modèle : {self.model.names}")
        self.model_name = model_path.name if model_path.exists() else det_cfg.fallback_model

    def detect(self, frame: np.ndarray):  # noqa: ANN201 - objet Results d'ultralytics
        """Inférence brute sur une image (résultat ultralytics)."""
        return self.model.predict(
            frame,
            conf=min(self.cfg.confidence, self.cfg.ball_confidence),
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            classes=list(self.class_map),
            verbose=False,
        )[0]

    def split(self, result) -> tuple[object, Detections]:  # noqa: ANN001
        """Sépare les personnes (à suivre) du ballon (meilleure détection, piste -1).

        Returns:
            Les boîtes ultralytics des personnes et les détections du ballon.
        """
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None, Detections()
        conf = boxes.conf.cpu().numpy()
        raw_cls = boxes.cls.cpu().numpy().astype(int)
        cls = np.array([self.class_map.get(int(c), -1) for c in raw_cls], dtype=int)
        is_ball = cls == ObjectClass.BALL
        people_mask = (cls >= 0) & ~is_ball & (conf >= self.cfg.confidence)
        ball = Detections()
        ball_idx = np.flatnonzero(is_ball & (conf >= self.cfg.ball_confidence))
        if len(ball_idx):
            best = int(ball_idx[np.argmax(conf[ball_idx])])
            ball = Detections(boxes.xyxy[best : best + 1].cpu().numpy(), conf[best : best + 1],
                              np.array([ObjectClass.BALL]), np.array([-1]))
        people = boxes[np.flatnonzero(people_mask)] if people_mask.any() else None
        return people, ball
