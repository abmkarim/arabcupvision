"""Suivi multi-objets des personnes avec les trackers ByteTrack / BoT-SORT d'ultralytics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arabcupvision.detection.detector import Detections


def _load_tracker_args(name: str) -> SimpleNamespace:
    from ultralytics.utils import IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    path = check_yaml(name)
    try:
        from ultralytics.utils import YAML

        data = YAML.load(path)
    except ImportError:  # anciennes versions d'ultralytics
        from ultralytics.utils import yaml_load

        data = yaml_load(path)
    return IterableSimpleNamespace(**data)


class MultiObjectTracker:
    """Associe des identifiants stables aux détections de personnes, image après image."""

    def __init__(self, tracker: str = "bytetrack.yaml", frame_rate: float = 25.0) -> None:
        args = _load_tracker_args(tracker)
        # Le tampon de pistes perdues est exprimé en images à 30 i/s : on l'adapte au débit.
        args.track_buffer = max(1, int(round(args.track_buffer * frame_rate / 30.0)))
        if args.tracker_type == "botsort":
            from ultralytics.trackers.bot_sort import BOTSORT

            self._tracker = BOTSORT(args)
        else:
            from ultralytics.trackers.byte_tracker import BYTETracker

            self._tracker = BYTETracker(args)

    def reset(self) -> None:
        """Oublie toutes les pistes."""
        self._tracker.reset()

    def update(self, boxes, class_map: dict[int, int], frame: np.ndarray) -> Detections:  # noqa: ANN001
        """Met à jour les pistes avec les boîtes ultralytics d'une image.

        Args:
            boxes: Boîtes ultralytics (``Boxes``) des personnes, ou ``None``.
            class_map: Classe du modèle → classe canonique.
            frame: Image courante (utilisée par BoT-SORT pour la compensation de mouvement).
        """
        if boxes is None:
            from ultralytics.engine.results import Boxes

            boxes = Boxes(np.empty((0, 6), dtype=np.float32), frame.shape[:2])
        out = self._tracker.update(boxes.cpu().numpy(), frame)
        if len(boxes) == 0:
            return Detections()
        if out is None or len(out) == 0:
            return Detections()
        out = np.asarray(out)
        cls = np.array([class_map.get(int(c), -1) for c in out[:, 6]], dtype=int)
        return Detections(out[:, :4].astype(float), out[:, 5].astype(float), cls,
                          out[:, 4].astype(int))
