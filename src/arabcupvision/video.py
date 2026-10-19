"""Lecture et écriture vidéo (OpenCV), avec ré-encodage H.264 optionnel."""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VideoError(RuntimeError):
    """Vidéo absente, illisible ou vide."""


@dataclass(frozen=True)
class VideoInfo:
    """Métadonnées d'une vidéo."""

    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        """Durée (s)."""
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def probe(path: str | Path) -> VideoInfo:
    """Lit les métadonnées d'une vidéo, ou lève ``VideoError``."""
    p = Path(path)
    if not p.exists():
        raise VideoError(f"Vidéo introuvable : {p}")
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise VideoError(f"Vidéo illisible : {p}")
        info = VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 25.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        ok, _ = cap.read()
        if not ok or info.width == 0:
            raise VideoError(f"Aucune image décodable : {p}")
        return info
    finally:
        cap.release()


def working_size(info: VideoInfo, resize_width: int) -> tuple[int, int, float]:
    """Taille de travail (largeur, hauteur paires) et facteur d'échelle appliqué."""
    scale = 1.0 if resize_width <= 0 or resize_width >= info.width else resize_width / info.width
    w = int(round(info.width * scale)) // 2 * 2
    h = int(round(info.height * scale)) // 2 * 2
    return w, h, scale


def iter_frames(
    path: str | Path,
    frame_skip: int = 1,
    max_seconds: float = 0.0,
    size: tuple[int, int] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Parcourt les images (indice d'origine, image BGR) en sautant ``frame_skip - 1`` images."""
    info = probe(path)
    limit = int(max_seconds * info.fps) if max_seconds > 0 else None
    cap = cv2.VideoCapture(str(path))
    idx = 0
    try:
        while True:
            if limit is not None and idx >= limit:
                break
            if idx % frame_skip != 0:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            if size is not None and (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            yield idx, frame
            idx += 1
    finally:
        cap.release()


class VideoWriter:
    """Écrit une vidéo MP4 ; ``close`` ré-encode en H.264 si ffmpeg est disponible."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], h264: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.h264 = h264
        self._raw = self.path.with_name(self.path.stem + ".raw.mp4") if h264 else self.path
        self._writer = cv2.VideoWriter(str(self._raw), cv2.VideoWriter_fourcc(*"mp4v"),
                                       max(fps, 1.0), size)
        if not self._writer.isOpened():
            raise VideoError(f"Impossible d'ouvrir la vidéo en écriture : {self.path}")
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        """Ajoute une image."""
        self._writer.write(frame)
        self.frames += 1

    def close(self) -> Path:
        """Termine l'écriture et renvoie le chemin final."""
        self._writer.release()
        if self.h264 and self._raw != self.path:
            if transcode_h264(self._raw, self.path):
                self._raw.unlink(missing_ok=True)
            else:
                shutil.move(str(self._raw), str(self.path))
        return self.path

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def ffmpeg_executable() -> str | None:
    """Chemin d'un binaire ffmpeg (imageio-ffmpeg, sinon PATH)."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - paquet optionnel
        return shutil.which("ffmpeg")


def transcode_h264(src: Path, dst: Path) -> bool:
    """Ré-encode ``src`` en H.264 (lisible par les navigateurs) ; ``False`` si échec."""
    exe = ffmpeg_executable()
    if exe is None:
        return False
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset",
           "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("Ré-encodage H.264 impossible (%s) ; vidéo MPEG-4 conservée.", exc)
        return False
    return dst.exists() and dst.stat().st_size > 0
