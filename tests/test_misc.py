import numpy as np
import pandas as pd

from arabcupvision.config import load_config
from arabcupvision.detection.detector import ObjectClass, canonical_class
from arabcupvision.evaluation.teams_eval import team_accuracy
from arabcupvision.evaluation.tracking_eval import tracks_to_mot
from arabcupvision.synthetic import generate_synthetic_match
from arabcupvision.video import probe


def test_config_loads_with_overrides():
    cfg = load_config(overrides={"video.frame_skip": 3})
    assert cfg.video.frame_skip == 3
    assert len(cfg.pitch.keypoints_m) == 32
    assert cfg.pitch.length_m == 105.0


def test_canonical_classes():
    assert canonical_class("ball") is ObjectClass.BALL
    assert canonical_class("sports ball") is ObjectClass.BALL
    assert canonical_class("goalkeeper") is ObjectClass.GOALKEEPER
    assert canonical_class("referee") is ObjectClass.REFEREE
    assert canonical_class("person") is ObjectClass.PLAYER
    assert canonical_class("car") is None


def test_synthetic_generator(tmp_path):
    paths = generate_synthetic_match(tmp_path / "s.mp4", seconds=0.4, fps=10, size=(320, 180))
    info = probe(paths["video"])
    assert info.frame_count == 4 and info.width == 320
    gt = pd.read_csv(paths["gt"], header=None)
    assert gt[0].min() == 1 and gt[1].nunique() == 19
    assert paths["calibration"].exists()


def test_team_accuracy_is_permutation_invariant():
    labels = pd.DataFrame({"track_id": [1, 2, 3, 4], "team": ["A", "A", "B", "B"]})
    pred = pd.DataFrame({"track_id": [1, 1, 2, 3, 4], "team": [1, 1, 1, 0, 1]})
    res = team_accuracy(labels, pred)
    assert res["accuracy"] == 0.75


def test_tracks_to_mot_scale():
    tracks = pd.DataFrame({"frame": [0], "track_id": [3], "x1": [10.0], "y1": [20.0],
                           "x2": [30.0], "y2": [60.0]})
    mot = tracks_to_mot(tracks, scale=0.5)
    np.testing.assert_allclose(mot.iloc[0].to_numpy(), [1, 3, 20, 40, 40, 80])


def test_iou_distance_and_tracking_metrics():
    from arabcupvision.evaluation.tracking_eval import evaluate_tracking, iou_distance

    d = iou_distance(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10], [5, 0, 10, 10],
                                                           [50, 50, 5, 5]]), 0.5)
    assert d[0, 0] == 0.0 and np.isnan(d[0, 1]) and np.isnan(d[0, 2])
    gt = pd.DataFrame({"frame": [1, 1, 2, 2], "id": [1, 2, 1, 2], "x": [0, 50, 2, 52],
                       "y": [0, 0, 0, 0], "w": [10, 10, 10, 10], "h": [20, 20, 20, 20]})
    hyp = gt.assign(id=gt["id"] + 10)
    m = evaluate_tracking(gt, hyp)
    assert m["idf1"] == 1.0 and m["mota"] == 1.0 and m["num_switches"] == 0
