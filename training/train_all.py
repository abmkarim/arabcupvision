"""Téléchargement des datasets Roboflow Universe et entraînements YOLOv8 plafonnés (CPU).

Usage : python training/train_all.py [--config training/train.yaml] [--only detector]
La clé ROBOFLOW_API_KEY est lue dans l'environnement ou dans le fichier .env (jamais affichée).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_api_key() -> str | None:
    """Lit ROBOFLOW_API_KEY dans l'environnement puis dans .env."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ROBOFLOW_API_KEY":
                key = value.strip().strip('"').strip("'")
    return key or None


def download(job: dict, api_key: str, datasets_dir: Path) -> Path:
    """Télécharge un dataset (une seule fois) et renvoie le chemin de son data.yaml."""
    target = datasets_dir / job["project"]
    data_yaml = target / "data.yaml"
    if not data_yaml.exists():
        from roboflow import Roboflow

        project = Roboflow(api_key=api_key).workspace(job["workspace"]).project(job["project"])
        version = job.get("version")
        if version is None:
            version = max(int(str(v.version).split("/")[-1]) for v in project.versions())
        project.version(int(version)).download(job["format"], location=str(target),
                                               overwrite=True)
    fix_data_yaml(data_yaml)
    return data_yaml


def fix_data_yaml(data_yaml: Path) -> None:
    """Rend les chemins du data.yaml absolus (format Roboflow → ultralytics)."""
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = data_yaml.parent
    data["path"] = str(base)
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        if (base / folder / "images").exists():
            data[split] = f"{folder}/images"
        else:
            data.pop(split, None)
    if "val" not in data:
        data["val"] = data["train"]
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def train(job: dict, data_yaml: Path, cfg: dict, name: str) -> Path:
    """Entraîne un modèle et copie ses meilleurs poids vers ``job['output']``."""
    from ultralytics import YOLO

    base = ROOT / "models" / job["base_model"]
    base.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(base))
    kwargs = dict(
        data=str(data_yaml), epochs=job["epochs"], imgsz=job["imgsz"], batch=job["batch"],
        device=cfg["device"], workers=cfg["workers"], seed=cfg["seed"], deterministic=True,
        project=str(ROOT / cfg["runs_dir"]), name=name, exist_ok=True, plots=False,
        verbose=False,
    )
    if job.get("time_hours"):
        kwargs["time"] = job["time_hours"]
    model.train(**kwargs)
    best = ROOT / cfg["runs_dir"] / name / "weights" / "best.pt"
    out = ROOT / job["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    return out


def main() -> int:
    """Point d'entrée."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "training" / "train.yaml"))
    parser.add_argument("--only", choices=["detector", "keypoints"], default=None)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if cfg_path.name == "default.yaml":  # appel générique via tasks.ps1
        cfg_path = ROOT / "training" / "train.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    api_key = read_api_key()
    if not api_key:
        print("ROBOFLOW_API_KEY absente : le pipeline utilisera le modèle COCO de repli.")
        return 1
    status = 0
    for name, job in cfg["jobs"].items():
        if args.only and name != args.only:
            continue
        try:
            data_yaml = download(job, api_key, ROOT / cfg["datasets_dir"])
            print(f"[{name}] dataset prêt : {data_yaml.parent.name}", flush=True)
            if not args.download_only:
                out = train(job, data_yaml, cfg, name)
                print(f"[{name}] poids enregistrés : {out.relative_to(ROOT)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - un échec ne doit pas bloquer l'autre job
            print(f"[{name}] échec ({type(exc).__name__}: {exc}) ; repli documenté utilisé.",
                  flush=True)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
