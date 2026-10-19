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
