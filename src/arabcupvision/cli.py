"""Interface en ligne de commande : ``python -m arabcupvision.cli <commande>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from arabcupvision.config import load_config


def _overrides(args: argparse.Namespace) -> dict:
    out: dict = {}
    if getattr(args, "max_seconds", None) is not None:
        out["video.max_seconds"] = args.max_seconds
    if getattr(args, "frame_skip", None) is not None:
        out["video.frame_skip"] = args.frame_skip
    if getattr(args, "resize_width", None) is not None:
        out["video.resize_width"] = args.resize_width
    if getattr(args, "model", None):
        out["detection.model_path"] = args.model
    return out


def _add_video_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="Fichier YAML (défaut : configs/default.yaml).")
    p.add_argument("--max-seconds", type=float, default=None, help="Durée maximale traitée.")
    p.add_argument("--frame-skip", type=int, default=None, help="Traiter une image sur N.")
    p.add_argument("--resize-width", type=int, default=None, help="Largeur de travail.")
    p.add_argument("--model", default=None, help="Poids du détecteur.")
    p.add_argument("--calibration", default=None, help="Calibrage manuel (JSON).")


def cmd_process(args: argparse.Namespace) -> int:
    """Traite une vidéo et écrit vidéo annotée, CSV, heatmaps et JSON."""
    from arabcupvision.pipeline import Pipeline

    cfg = load_config(args.config, _overrides(args))
    last = [-1]

    def progress(frac: float, msg: str) -> None:
        pct = int(frac * 100)
        if pct // 10 != last[0] // 10:
            last[0] = pct
            print(f"[{pct:3d}%] {msg}", flush=True)

    result = Pipeline(cfg, args.calibration, progress).run(args.video, args.output_dir)
    print(f"Sorties : {result.output_dir}")
    print(json.dumps(result.summary["outputs"], indent=2, ensure_ascii=False))
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    """Génère la vidéo synthétique de test."""
    from arabcupvision.synthetic import generate_synthetic_match

    paths = generate_synthetic_match(args.output, seconds=args.seconds, fps=args.fps,
                                     seed=args.seed)
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


def cmd_eval_tracking(args: argparse.Namespace) -> int:
    """IDF1 / MOTA / ID switches à partir d'une vérité terrain MOTChallenge."""
    import pandas as pd

    from arabcupvision.evaluation.tracking_eval import evaluate_tracking, load_mot, tracks_to_mot
    from arabcupvision.video import probe, working_size

    cfg = load_config(args.config, _overrides(args))
    if args.predictions:
        tracks = pd.read_csv(args.predictions)
    else:
        if not args.video:
            print("Précisez --video ou --predictions.", file=sys.stderr)
            return 2
        from arabcupvision.pipeline import Pipeline

        tracks = Pipeline(cfg, args.calibration).run(args.video, args.output_dir,
                                                     render=False).tracks
    scale = 1.0
    if args.video:
        _, _, scale = working_size(probe(args.video), cfg.video.resize_width)
    hyp = tracks_to_mot(tracks, scale)
    metrics = evaluate_tracking(load_mot(args.gt), hyp, args.iou)
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_eval_teams(args: argparse.Namespace) -> int:
    """Précision de l'assignation d'équipe."""
    import pandas as pd

    from arabcupvision.evaluation.teams_eval import load_team_labels, team_accuracy

    metrics = team_accuracy(load_team_labels(args.labels), pd.read_csv(args.predictions))
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Mesure le débit (images/s) du pipeline et de chacune de ses étapes."""
    from arabcupvision.pipeline import Pipeline

    cfg = load_config(args.config, _overrides(args))
    start = time.perf_counter()
    result = Pipeline(cfg, args.calibration).run(args.video, args.output_dir,
                                                 render=not args.no_render)
    total = time.perf_counter() - start
    n = result.summary["frames_processed"]
    report = {
        "frames_processed": n,
        "total_s": round(total, 2),
        "fps_total": round(n / total, 2) if total > 0 else 0.0,
        "stages_s": {k: round(v, 2) for k, v in result.timings.items()},
    }
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments."""
    parser = argparse.ArgumentParser(prog="arabcupvision", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="Analyser une vidéo de match.")
    p.add_argument("video")
    p.add_argument("--output-dir", default=None)
    _add_video_options(p)
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("synth", help="Générer une vidéo synthétique de test.")
    p.add_argument("--output", default="data/videos/synthetic_demo.mp4")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("eval-tracking", help="IDF1 / MOTA / ID switches (motmetrics).")
    p.add_argument("--gt", required=True, help="Vérité terrain MOTChallenge (gt.txt).")
    p.add_argument("--video", default=None, help="Vidéo à traiter (si pas de --predictions).")
    p.add_argument("--predictions", default=None, help="tracks.csv déjà produit.")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--output-dir", default=None)
    _add_video_options(p)
    p.set_defaults(func=cmd_eval_tracking)

    p = sub.add_parser("eval-teams", help="Précision de l'assignation d'équipe.")
    p.add_argument("--labels", required=True, help="CSV track_id,team.")
    p.add_argument("--predictions", required=True, help="tracks.csv ou players.csv.")
    p.set_defaults(func=cmd_eval_teams)

    p = sub.add_parser("benchmark", help="Mesure du débit du pipeline.")
    p.add_argument("video")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-render", action="store_true")
    _add_video_options(p)
    p.set_defaults(func=cmd_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
