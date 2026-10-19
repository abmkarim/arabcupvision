"""Configuration unique du projet, chargée depuis un fichier YAML et validée par pydantic."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class VideoConfig(BaseModel):
    """Lecture de la vidéo source."""

    frame_skip: int = Field(1, ge=1, description="Traiter une image sur N.")
    max_seconds: float = Field(0.0, ge=0.0, description="Durée maximale traitée (0 = tout).")
    resize_width: int = Field(1280, ge=0, description="Largeur de travail (0 = native).")


class DetectionConfig(BaseModel):
    """Modèle de détection des joueurs, gardiens, arbitres et du ballon."""

    model_path: str = "models/detector.pt"
    fallback_model: str = "yolov8n.pt"
    confidence: float = Field(0.25, ge=0.0, le=1.0)
    ball_confidence: float = Field(0.10, ge=0.0, le=1.0)
    iou: float = Field(0.5, ge=0.0, le=1.0)
    imgsz: int = Field(960, ge=32)
    device: str = "cpu"


class TrackingConfig(BaseModel):
    """Suivi multi-objets et post-traitement des trajectoires."""

    tracker: str = "bytetrack.yaml"
    smoothing_window: int = Field(5, ge=1, description="Fenêtre de lissage (images traitées).")
    ball_max_gap: int = Field(15, ge=0, description="Trou maximal interpolé pour le ballon.")
    min_track_length: int = Field(3, ge=1, description="Pistes plus courtes ignorées.")


class TeamsConfig(BaseModel):
    """Assignation des équipes par couleur de maillot."""

    fit_frames: int = Field(30, ge=1, description="Images utilisées pour ajuster le K-Means.")
    crop_top: float = Field(0.15, ge=0.0, le=1.0)
    crop_bottom: float = Field(0.50, ge=0.0, le=1.0)
    crop_side: float = Field(0.25, ge=0.0, lt=0.5)
    green_hue_range: tuple[int, int] = (35, 85)
    green_min_saturation: int = 40
    min_pixels: int = 20
    outlier_factor: float = Field(
        2.5, gt=0.0, description="Distance relative au-delà de laquelle une couleur est atypique."
    )
    min_outlier_distance: float = Field(25.0, gt=0.0, description="Distance Lab minimale.")


class PitchConfig(BaseModel):
    """Terrain réglementaire et calcul d'homographie."""

    length_m: float = 105.0
    width_m: float = 68.0
    keypoint_model_path: str = "models/pitch_keypoints.pt"
    keypoint_confidence: float = Field(0.5, ge=0.0, le=1.0)
    min_keypoints: int = Field(4, ge=4)
    recompute_interval: int = Field(1, ge=1, description="Recalcul toutes les N images traitées.")
    ransac_threshold_m: float = 1.0
    keypoints_m: list[tuple[float, float]] = Field(
        default_factory=list,
        description="Coordonnées (m) des points clés du modèle, dans l'ordre de ses indices.",
    )


class AnalyticsConfig(BaseModel):
    """Statistiques par joueur."""

    max_speed_kmh: float = Field(40.0, gt=0.0, description="Seuil anti-aberrations.")
    speed_window: int = Field(5, ge=1)
    heatmap_bins: tuple[int, int] = (42, 27)
    heatmap_sigma: float = 1.5
    heatmap_min_frames: int = Field(10, ge=1, description="Heatmap si la piste est assez longue.")
    possession_max_distance_m: float = 3.0
    possession_max_distance_px: float = 60.0


class RenderConfig(BaseModel):
    """Annotation de la vidéo et mini-carte."""

    minimap: bool = True
    minimap_width: int = 360
    minimap_alpha: float = Field(0.85, ge=0.0, le=1.0)
    team_colors_bgr: tuple[tuple[int, int, int], tuple[int, int, int]] = (
        (60, 60, 230),
        (230, 160, 40),
    )
    referee_color_bgr: tuple[int, int, int] = (0, 215, 255)
    unknown_color_bgr: tuple[int, int, int] = (180, 180, 180)
    ball_color_bgr: tuple[int, int, int] = (255, 255, 255)
    h264: bool = True


class OutputConfig(BaseModel):
    """Emplacement des sorties."""

    root: str = "outputs"


class Config(BaseModel):
    """Configuration complète du pipeline."""

    seed: int = 42
    video: VideoConfig = VideoConfig()
    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    teams: TeamsConfig = TeamsConfig()
    pitch: PitchConfig = PitchConfig()
    analytics: AnalyticsConfig = AnalyticsConfig()
    render: RenderConfig = RenderConfig()
    output: OutputConfig = OutputConfig()

    def resolve(self, path: str | Path) -> Path:
        """Résout un chemin relatif par rapport à la racine du projet."""
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Charge la configuration YAML et applique d'éventuelles surcharges imbriquées."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    for key, value in (overrides or {}).items():
        section, _, field = key.partition(".")
        if field:
            data.setdefault(section, {})[field] = value
        else:
            data[section] = value
    return Config.model_validate(data)


def seed_everything(seed: int) -> None:
    """Fixe les graines aléatoires utilisées par le projet."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
