"""Application Streamlit : import d'une vidéo, traitement et exploration des statistiques."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arabcupvision.analytics.heatmap import heatmap_grid, render_heatmap  # noqa: E402
from arabcupvision.config import load_config  # noqa: E402
from arabcupvision.pitch.geometry import PitchDimensions  # noqa: E402
from arabcupvision.video import VideoError, probe  # noqa: E402

TEAM_LABELS = {0: "Équipe A", 1: "Équipe B", -1: "Inconnue"}


def available_models() -> list[str]:
    """Poids disponibles dans models/ (détecteurs uniquement) + repli COCO."""
    models_dir = ROOT / "models"
    found = []
    if models_dir.exists():
        found = sorted(p.name for p in models_dir.glob("*.pt")
                       if "pose" not in p.name and "keypoint" not in p.name)
    names = [f"models/{n}" for n in found]
    if "models/detector.pt" in names:
        names.remove("models/detector.pt")
        names.insert(0, "models/detector.pt")
    if not any(n.endswith("yolov8n.pt") for n in names):
        names.append("models/yolov8n.pt")
    return names


def sidebar_settings(cfg) -> dict:  # noqa: ANN001
    """Réglages du traitement."""
    st.sidebar.header("Réglages")
    model = st.sidebar.selectbox("Modèle de détection", available_models(),
                                 help="detector.pt = modèle fine-tuné ; yolov8n.pt = repli COCO.")
    frame_skip = st.sidebar.slider("Pas d'images (1 = toutes)", 1, 10, cfg.video.frame_skip)
    resize = st.sidebar.select_slider("Largeur de travail (px)", [640, 960, 1280, 1920],
                                      value=cfg.video.resize_width
                                      if cfg.video.resize_width in (640, 960, 1280, 1920)
                                      else 1280)
    max_seconds = st.sidebar.number_input("Durée maximale traitée (s, 0 = tout)", 0.0, 3600.0,
                                          20.0, step=5.0)
    return {"detection.model_path": model, "video.frame_skip": frame_skip,
            "video.resize_width": resize, "video.max_seconds": max_seconds}


def run_pipeline(video_path: Path, overrides: dict, work_dir: Path):  # noqa: ANN201
    """Lance le pipeline avec une barre de progression."""
    from arabcupvision.pipeline import Pipeline

    cfg = load_config(overrides=overrides)
    bar = st.progress(0.0, text="Initialisation…")

    def progress(frac: float, msg: str) -> None:
        bar.progress(min(max(frac, 0.0), 1.0), text=msg)

    result = Pipeline(cfg, progress=progress).run(video_path, work_dir / "results")
    bar.progress(1.0, text="Terminé")
    return result


def show_results(result) -> None:  # noqa: ANN001
    """Affiche vidéo, tableau, heatmaps et graphiques."""
    summary = result.summary
    players: pd.DataFrame = result.players.copy()
    tracks: pd.DataFrame = result.tracks

    c1, c2, c3 = st.columns(3)
    c1.metric("Images traitées", summary["frames_processed"])
    poss = summary["possession_pct"]
    c2.metric("Possession A / B", f"{poss['team_0']:.0f} % / {poss['team_1']:.0f} %")
    c3.metric("Homographie", ", ".join(summary["homography_sources"]))
    if summary["detector_fallback_coco"]:
        st.info("Modèle fine-tuné absent : détection COCO de repli (arbitres et gardiens "
                "repérés par la couleur).")

    st.subheader("Vidéo annotée")
    if result.video and result.video.exists():
        data = result.video.read_bytes()
        st.video(data)
        st.download_button("Télécharger la vidéo", data, file_name="annotated.mp4",
                           mime="video/mp4")

    st.subheader("Statistiques par joueur")
    if players.empty:
        st.warning("Aucun joueur suivi suffisamment longtemps.")
        return
    players.insert(1, "équipe", players["team"].map(TEAM_LABELS))
    teams = st.multiselect("Filtrer par équipe", sorted(players["équipe"].unique()),
                           default=sorted(players["équipe"].unique()))
    view = players[players["équipe"].isin(teams)]
    sort_col = st.selectbox("Trier par", ["distance_m", "max_speed_kmh", "avg_speed_kmh",
                                          "presence_s", "track_id"])
    view = view.sort_values(sort_col, ascending=sort_col == "track_id")
    st.dataframe(view.drop(columns=["team"]), use_container_width=True, hide_index=True)
    st.download_button("Télécharger le CSV", view.to_csv(index=False).encode("utf-8"),
                       file_name="players.csv", mime="text/csv")

    g1, g2 = st.columns(2)
    chart = view.set_index(view["track_id"].astype(str))
    g1.caption("Distance parcourue (m)")
    g1.bar_chart(chart["distance_m"])
    g2.caption("Vitesse maximale (km/h)")
    g2.bar_chart(chart["max_speed_kmh"])

    st.subheader("Heatmap d'un joueur")
    tid = st.selectbox("Piste", view["track_id"].tolist())
    sel = tracks[tracks["track_id"] == tid]
    dims = PitchDimensions()
    cfg = load_config()
    grid = heatmap_grid(sel[["pitch_x", "pitch_y"]].to_numpy(), dims.length, dims.width,
                        cfg.analytics.heatmap_bins, cfg.analytics.heatmap_sigma)
    team = int(view.loc[view["track_id"] == tid, "team"].iat[0])
    st.pyplot(render_heatmap(grid, dims, f"Piste {tid} – {TEAM_LABELS[team]}"))
    speed = sel[["time_s", "speed_kmh"]].dropna().set_index("time_s")
    if not speed.empty:
        st.caption("Vitesse lissée (km/h) au cours du temps")
        st.line_chart(speed)


def main() -> None:
    """Page principale."""
    st.set_page_config(page_title="ArabCupVision", layout="wide")
    st.title("ArabCupVision")
    st.caption("Détection, suivi, équipes, mini-carte tactique et statistiques par joueur.")
    cfg = load_config()
    overrides = sidebar_settings(cfg)

    upload = st.file_uploader("Vidéo de match", type=["mp4", "avi", "mov", "mkv"])
    if upload is None:
        st.info("Importez une vidéo pour commencer.")
        return
    if "work_dir" not in st.session_state:
        st.session_state.work_dir = tempfile.mkdtemp(prefix="arabcupvision_")
    work_dir = Path(st.session_state.work_dir)
    video_path = work_dir / Path(upload.name).name
    video_path.write_bytes(upload.getbuffer())
    try:
        info = probe(video_path)
    except VideoError as exc:
        st.error(f"Vidéo illisible : {exc}")
        return
    st.write(f"{info.width}×{info.height}, {info.fps:.1f} i/s, {info.duration_s:.1f} s")

    if st.button("Lancer l'analyse", type="primary"):
        start = time.perf_counter()
        try:
            st.session_state.result = run_pipeline(video_path, overrides, work_dir)
            st.success(f"Analyse terminée en {time.perf_counter() - start:.0f} s.")
        except VideoError as exc:
            st.error(f"Problème de vidéo : {exc}")
        except FileNotFoundError as exc:
            st.error(f"Fichier ou modèle introuvable : {exc}")
        except Exception as exc:  # noqa: BLE001 - affichage propre de toute erreur
            st.error(f"Échec du traitement : {type(exc).__name__}: {exc}")
    if "result" in st.session_state:
        show_results(st.session_state.result)


if __name__ == "__main__":
    main()
