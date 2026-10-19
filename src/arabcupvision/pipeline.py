"""Pipeline complet : détection, suivi, équipes, homographie, statistiques et rendu."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from arabcupvision.analytics.heatmap import heatmap_grid, render_heatmap
from arabcupvision.analytics.kinematics import player_statistics
from arabcupvision.analytics.possession import possession_per_frame, possession_share
from arabcupvision.config import Config, seed_everything
from arabcupvision.detection.detector import Detector, ObjectClass
from arabcupvision.pitch.geometry import PitchDimensions
from arabcupvision.pitch.homography import (
    HomographyTracker,
    approximate_homography,
    load_calibration,
    transform_points,
)
from arabcupvision.rendering.annotator import (
    FrameItem,
    MiniMap,
    draw_ball,
    draw_header,
    draw_player,
)
from arabcupvision.teams.classifier import (
    UNKNOWN_TEAM,
    TeamClassifier,
    assign_by_centroid,
    majority_vote,
)
from arabcupvision.teams.colors import jersey_color, lab_to_bgr
from arabcupvision.tracking.postprocess import interpolate_ball, smooth_tracks
from arabcupvision.tracking.tracker import MultiObjectTracker
from arabcupvision.video import VideoError, VideoWriter, iter_frames, probe, working_size

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

ROLE_NAMES = {
    ObjectClass.PLAYER: "player",
    ObjectClass.GOALKEEPER: "goalkeeper",
    ObjectClass.REFEREE: "referee",
}


@dataclass
class PipelineResult:
    """Chemins des sorties et tableaux produits par le pipeline."""

    output_dir: Path
    video: Path | None
    tracks_csv: Path
    players_csv: Path
    ball_csv: Path
    summary_json: Path
    heatmaps: list[Path]
    players: pd.DataFrame
    tracks: pd.DataFrame
    summary: dict
    timings: dict[str, float] = field(default_factory=dict)


class Pipeline:
    """Analyse d'une vidéo de match en deux passes (analyse, puis rendu)."""

    def __init__(
        self,
        cfg: Config,
        calibration: str | Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.cfg = cfg
        self.calibration = Path(calibration) if calibration else None
        self.progress = progress or (lambda frac, msg: None)
        self.dims = PitchDimensions(length=cfg.pitch.length_m, width=cfg.pitch.width_m)
        self.timings: dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------ chargement
    def _load_detector(self) -> Detector:
        root = self.cfg.resolve(".")
        return Detector(self.cfg.detection, self.cfg.tracking, root)

    def _load_keypoints(self):  # noqa: ANN202
        path = self.cfg.resolve(self.cfg.pitch.keypoint_model_path)
        if not path.exists() or not self.cfg.pitch.keypoints_m:
            return None
        from arabcupvision.pitch.keypoints import PitchKeypointDetector

        try:
            return PitchKeypointDetector(path, self.cfg.pitch, self.cfg.detection.device)
        except Exception as exc:  # noqa: BLE001 - un modèle défectueux ne doit pas tout bloquer
            log.warning("Modèle de points clés inutilisable (%s).", exc)
            return None

    # ------------------------------------------------------------------ exécution
    def run(
        self,
        video_path: str | Path,
        output_dir: str | Path | None = None,
        render: bool = True,
    ) -> PipelineResult:
        """Traite une vidéo et écrit toutes les sorties dans ``output_dir``."""
        seed_everything(self.cfg.seed)
        video_path = Path(video_path)
        info = probe(video_path)
        size = working_size(info, self.cfg.video.resize_width)
        out_dir = Path(output_dir) if output_dir else self.cfg.resolve(
            Path(self.cfg.output.root) / video_path.stem
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        tracks, ball, h_sources, n_frames, frames = self._analyse(video_path, info, size)
        self.timings["analysis"] = time.perf_counter() - t0
        if n_frames == 0:
            raise VideoError(f"Aucune image traitée : {video_path}")

        t1 = time.perf_counter()
        tracks, team_colors, fallback_roles = self._assign_teams(tracks)
        tracks = smooth_tracks(tracks, ["pitch_x", "pitch_y"], self.cfg.tracking.smoothing_window)
        ball_full = interpolate_ball(ball, frames, self.cfg.tracking.ball_max_gap,
                                     columns=("x", "y", "pitch_x", "pitch_y"))
        frame_dt = self.cfg.video.frame_skip / info.fps
        movers = tracks[tracks["role"] != "referee"]
        players, speeds = player_statistics(movers, self.cfg.analytics.max_speed_kmh,
                                            self.cfg.analytics.speed_window, frame_dt)
        tracks["speed_kmh"] = speeds.reindex(tracks.index)
        poss = possession_per_frame(
            tracks[tracks["role"] != "referee"],
            ball_full.dropna(subset=["pitch_x"]),
            self.cfg.analytics.possession_max_distance_m,
        )
        heatmaps = self._heatmaps(tracks, players, out_dir / "heatmaps")
        self.timings["analytics"] = time.perf_counter() - t1

        video_out: Path | None = None
        if render:
            t2 = time.perf_counter()
            video_out = self._render(video_path, info, size, tracks, ball_full, poss,
                                     out_dir / "annotated.mp4")
            self.timings["render"] = time.perf_counter() - t2

        tracks_csv = out_dir / "tracks.csv"
        players_csv = out_dir / "players.csv"
        ball_csv = out_dir / "ball.csv"
        tracks.sort_values(["frame", "track_id"]).to_csv(tracks_csv, index=False)
        players.to_csv(players_csv, index=False)
        ball_full.merge(poss, on="frame", how="left").to_csv(ball_csv, index=False)

        summary = {
            "video": video_path.name,
            "source_resolution": [info.width, info.height],
            "working_resolution": list(size[:2]),
            "source_fps": round(info.fps, 3),
            "frame_skip": self.cfg.video.frame_skip,
            "frames_processed": n_frames,
            "duration_processed_s": round(n_frames * frame_dt, 2),
            "detector": self._detector_name,
            "detector_fallback_coco": self._detector_fallback,
            "fallback_roles_by_color": fallback_roles,
            "homography_sources": h_sources,
            "team_colors_bgr": team_colors,
            "players_per_team": {
                f"team_{t}": int((players["team"] == t).sum()) if not players.empty else 0
                for t in (0, 1)
            },
            "possession_pct": possession_share(poss),
            "ball_detected_frames": int(ball["frame"].nunique()) if not ball.empty else 0,
            "outputs": {
                "video": video_out.name if video_out else None,
                "tracks": tracks_csv.name,
                "players": players_csv.name,
                "ball": ball_csv.name,
                "heatmaps": [p.relative_to(out_dir).as_posix() for p in heatmaps],
            },
        }
        summary_json = out_dir / "summary.json"
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                encoding="utf-8")
        self.progress(1.0, "Terminé")
        return PipelineResult(out_dir, video_out, tracks_csv, players_csv, ball_csv,
                              summary_json, heatmaps, players, tracks, summary,
                              dict(self.timings))

    # ------------------------------------------------------------------ passe 1
    def _analyse(self, video_path: Path, info, size):  # noqa: ANN001, ANN202
        cfg = self.cfg
        w, h, scale = size
        detector = self._load_detector()
        self._detector_name = detector.model_name
        self._detector_fallback = detector.fallback
        tracker = MultiObjectTracker(cfg.tracking.tracker, info.fps / cfg.video.frame_skip)
        keypoints = self._load_keypoints()
        static_H, static_source = self._static_homography(video_path, w, h, scale)
        hom = HomographyTracker(cfg.pitch.recompute_interval)
        h_sources: dict[str, int] = defaultdict(int)

        total = info.frame_count
        if cfg.video.max_seconds > 0:
            total = min(total, int(cfg.video.max_seconds * info.fps))
        expected = max(1, total // cfg.video.frame_skip)

        records: list[dict] = []
        ball_rows: list[dict] = []
        frames: list[int] = []
        for i, (idx, frame) in enumerate(iter_frames(video_path, cfg.video.frame_skip,
                                                     cfg.video.max_seconds, (w, h))):
            frames.append(idx)
            t_s = idx / info.fps
            t = time.perf_counter()
            result = detector.detect(frame)
            people_boxes, ball_det = detector.split(result)
            self.timings["detection"] += time.perf_counter() - t

            t = time.perf_counter()
            people = tracker.update(people_boxes, detector.class_map, frame)
            self.timings["tracking"] += time.perf_counter() - t

            t = time.perf_counter()
            if keypoints is not None and hom.should_update():
                hom.update(keypoints.homography(frame), "keypoints")
            H, src = (hom.current, hom.source) if hom.current is not None else (static_H,
                                                                                static_source)
            h_sources[src] += 1
            self.timings["homography"] += time.perf_counter() - t

            t = time.perf_counter()
            feet = people.foot_points
            pitch = transform_points(H, feet) if len(feet) else np.empty((0, 2))
            for j in range(len(people)):
                color = jersey_color(frame, people.xyxy[j], cfg.teams)
                x1, y1, x2, y2 = people.xyxy[j]
                records.append({
                    "frame": idx, "time_s": round(t_s, 4), "track_id": int(people.track_id[j]),
                    "class_id": int(people.class_id[j]),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": float(people.confidence[j]),
                    "pitch_x": float(pitch[j, 0]), "pitch_y": float(pitch[j, 1]),
                    "color_l": np.nan if color is None else float(color[0]),
                    "color_a": np.nan if color is None else float(color[1]),
                    "color_b": np.nan if color is None else float(color[2]),
                })
            if len(ball_det):
                bx = float((ball_det.xyxy[0, 0] + ball_det.xyxy[0, 2]) / 2)
                by = float((ball_det.xyxy[0, 1] + ball_det.xyxy[0, 3]) / 2)
                bp = transform_points(H, np.array([[bx, float(ball_det.xyxy[0, 3])]]))[0]
                ball_rows.append({"frame": idx, "x": bx, "y": by, "pitch_x": float(bp[0]),
                                  "pitch_y": float(bp[1])})
            self.timings["teams_features"] += time.perf_counter() - t
            self.progress(0.75 * min(1.0, (i + 1) / expected),
                          f"Analyse : image {i + 1}/{expected}")

        columns = ["frame", "time_s", "track_id", "class_id", "x1", "y1", "x2", "y2",
                   "confidence", "pitch_x", "pitch_y", "color_l", "color_a", "color_b"]
        tracks = pd.DataFrame(records, columns=columns)
        if not tracks.empty:
            lengths = tracks.groupby("track_id")["frame"].transform("size")
            tracks = tracks[lengths >= cfg.tracking.min_track_length].reset_index(drop=True)
        ball = pd.DataFrame(ball_rows, columns=["frame", "x", "y", "pitch_x", "pitch_y"])
        return tracks, ball, dict(h_sources), len(frames), frames

    def _static_homography(self, video_path: Path, w: int, h: int,
                           scale: float) -> tuple[np.ndarray, str]:
        candidates = [self.calibration, video_path.with_suffix(".calibration.json")]
        for cand in candidates:
            if cand is not None and cand.exists():
                try:
                    return load_calibration(cand, scale), "calibration"
                except (ValueError, KeyError, OSError) as exc:
                    log.warning("Calibrage ignoré (%s).", exc)
        return approximate_homography(w, h, self.dims.length, self.dims.width), "approx"

    # ------------------------------------------------------------------ équipes
    def _assign_teams(self, tracks: pd.DataFrame) -> tuple[pd.DataFrame, list, bool]:
        cfg = self.cfg
        tracks = tracks.copy()
        tracks["role"] = tracks["class_id"].map(
            lambda c: ROLE_NAMES.get(ObjectClass(c), "player") if c >= 0 else "player"
        )
        tracks["team"] = UNKNOWN_TEAM
        if tracks.empty:
            return tracks, [], False
        color_cols = ["color_l", "color_a", "color_b"]
        has_color = tracks[color_cols].notna().all(axis=1)
        players = tracks[(tracks["role"] == "player") & has_color]
        first_frames = sorted(players["frame"].unique())[: cfg.teams.fit_frames]
        fit_set = players[players["frame"].isin(first_frames)]
        if len(fit_set) < 2:
            fit_set = players
        clf = TeamClassifier(cfg.seed, cfg.teams.outlier_factor, cfg.teams.min_outlier_distance)
        if len(fit_set) < 2:
            log.warning("Pas assez de couleurs de maillot pour séparer les équipes.")
            return tracks, [], False
        clf.fit(fit_set[color_cols].to_numpy())

        labels = np.full(len(tracks), UNKNOWN_TEAM)
        outlier = np.zeros(len(tracks), dtype=bool)
        colored = has_color.to_numpy()
        labels[colored] = clf.predict(tracks.loc[has_color, color_cols].to_numpy())
        outlier[colored] = clf.is_outlier(tracks.loc[has_color, color_cols].to_numpy())
        tracks["team_raw"] = labels

        fallback = bool(self._detector_fallback)
        if fallback:
            # Modèle COCO : arbitres et gardiens repérés par une couleur atypique. Une piste
            # atypique située derrière tous les joueurs de champ est un gardien, sinon un arbitre.
            tracks["_outlier"] = outlier
            share = tracks.groupby("track_id")["_outlier"].mean()
            mean_x = tracks.groupby("track_id")["pitch_x"].mean()
            odd = share[share > 0.5].index
            field_x = mean_x.drop(index=odd)
            lo, hi = (field_x.min(), field_x.max()) if len(field_x) else (np.inf, -np.inf)
            for tid in odd:
                behind = mean_x[tid] < lo or mean_x[tid] > hi
                tracks.loc[tracks["track_id"] == tid, "role"] = (
                    "goalkeeper" if behind else "referee"
                )
            tracks = tracks.drop(columns="_outlier")

        is_player = tracks["role"] == "player"
        votes = majority_vote(tracks.loc[is_player, "track_id"],
                              tracks.loc[is_player, "team_raw"])
        tracks.loc[is_player, "team"] = tracks.loc[is_player, "track_id"].map(votes).fillna(
            UNKNOWN_TEAM).astype(int)

        assigned = tracks[(tracks["team"] >= 0) & is_player]
        centroids = {int(t): g[["pitch_x", "pitch_y"]].mean().to_numpy()
                     for t, g in assigned.groupby("team")}
        gk = tracks[tracks["role"] == "goalkeeper"]
        gk_pos = {int(t): g[["pitch_x", "pitch_y"]].mean().to_numpy()
                  for t, g in gk.groupby("track_id")}
        for tid, team in assign_by_centroid(gk_pos, centroids).items():
            tracks.loc[tracks["track_id"] == tid, "team"] = team
        tracks["team"] = tracks["team"].astype(int)
        team_colors = [list(lab_to_bgr(c)) for c in clf.centers]
        return tracks, team_colors, fallback

    # ------------------------------------------------------------------ heatmaps
    def _heatmaps(self, tracks: pd.DataFrame, players: pd.DataFrame, folder: Path) -> list[Path]:
        cfg = self.cfg.analytics
        paths: list[Path] = []
        if players.empty:
            return paths
        folder.mkdir(parents=True, exist_ok=True)
        for team in (0, 1):
            pts = tracks.loc[tracks["team"] == team, ["pitch_x", "pitch_y"]].to_numpy()
            if len(pts):
                grid = heatmap_grid(pts, self.dims.length, self.dims.width, cfg.heatmap_bins,
                                    cfg.heatmap_sigma)
                p = folder / f"team_{team}.png"
                render_heatmap(grid, self.dims, f"Équipe {team}", p)
                paths.append(p)
        for row in players.itertuples():
            if row.frames < cfg.heatmap_min_frames:
                continue
            pts = tracks.loc[tracks["track_id"] == row.track_id, ["pitch_x", "pitch_y"]]
            grid = heatmap_grid(pts.to_numpy(), self.dims.length, self.dims.width,
                                cfg.heatmap_bins, cfg.heatmap_sigma)
            p = folder / f"player_{row.track_id}.png"
            render_heatmap(grid, self.dims, f"Piste {row.track_id} (équipe {row.team})", p)
            paths.append(p)
        return paths

    # ------------------------------------------------------------------ passe 2
    def _color_for(self, role: str, team: int) -> tuple[int, int, int]:
        rc = self.cfg.render
        if role == "referee":
            return tuple(rc.referee_color_bgr)
        if team in (0, 1):
            return tuple(rc.team_colors_bgr[team])
        return tuple(rc.unknown_color_bgr)

    def _render(self, video_path: Path, info, size, tracks: pd.DataFrame,  # noqa: ANN001
                ball: pd.DataFrame, poss: pd.DataFrame, out_path: Path) -> Path:
        cfg = self.cfg
        w, h, _ = size
        by_frame = dict(tuple(tracks.groupby("frame")))
        ball_by_frame = ball.set_index("frame")
        poss_by_frame = poss.set_index("frame")["team"] if not poss.empty else pd.Series(
            dtype=int)
        counts = {0: 0, 1: 0}
        minimap = MiniMap(self.dims, cfg.render) if cfg.render.minimap else None
        fps_out = info.fps / cfg.video.frame_skip
        expected = max(1, len(ball_by_frame))
        writer = VideoWriter(out_path, fps_out, (w, h), cfg.render.h264)
        try:
            for i, (idx, frame) in enumerate(iter_frames(video_path, cfg.video.frame_skip,
                                                         cfg.video.max_seconds, (w, h))):
                items: list[FrameItem] = []
                g = by_frame.get(idx)
                if g is not None:
                    for r in g.itertuples():
                        items.append(FrameItem(
                            np.array([r.x1, r.y1, r.x2, r.y2]), int(r.track_id),
                            self._color_for(r.role, int(r.team)), (r.pitch_x, r.pitch_y),
                            label="REF" if r.role == "referee" else str(int(r.track_id)),
                        ))
                for it in items:
                    draw_player(frame, it)
                ball_xy = None
                if idx in ball_by_frame.index:
                    b = ball_by_frame.loc[idx]
                    if np.isfinite(b["x"]):
                        draw_ball(frame, (b["x"], b["y"]), cfg.render.ball_color_bgr,
                                  bool(b["interpolated"]))
                        ball_xy = (b["pitch_x"], b["pitch_y"])
                team_now = int(poss_by_frame.get(idx, -1))
                if team_now in counts:
                    counts[team_now] += 1
                total = counts[0] + counts[1]
                header = f"t = {idx / info.fps:5.1f} s"
                if total:
                    header += (f"   possession  A {100 * counts[0] / total:3.0f}%"
                               f"  B {100 * counts[1] / total:3.0f}%")
                draw_header(frame, header)
                if minimap is not None:
                    minimap.overlay(frame, minimap.render(items, ball_xy))
                writer.write(frame)
                self.progress(0.75 + 0.25 * min(1.0, (i + 1) / expected),
                              f"Rendu : image {i + 1}/{expected}")
        finally:
            path = writer.close()
        return path
