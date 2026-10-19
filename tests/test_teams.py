import cv2
import numpy as np

from arabcupvision.config import TeamsConfig
from arabcupvision.teams.classifier import (
    UNKNOWN_TEAM,
    TeamClassifier,
    assign_by_centroid,
    majority_vote,
)
from arabcupvision.teams.colors import dominant_color_lab, jersey_color


def _lab(bgr):
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)


def test_kmeans_separates_two_kits(rng):
    red, white = _lab((40, 40, 200)), _lab((240, 240, 240))
    X = np.vstack([red + rng.normal(0, 3, (40, 3)), white + rng.normal(0, 3, (40, 3))])
    clf = TeamClassifier(seed=0).fit(X)
    pred = clf.predict(X)
    assert len(set(pred[:40])) == 1 and len(set(pred[40:])) == 1
    assert pred[0] != pred[40]


def test_outliers_do_not_capture_a_team(rng):
    red, white, black = _lab((40, 40, 200)), _lab((240, 240, 240)), _lab((20, 20, 20))
    X = np.vstack([red + rng.normal(0, 2, (6, 3)), white + rng.normal(0, 2, (20, 3)),
                   black + rng.normal(0, 2, (3, 3))])
    clf = TeamClassifier(seed=0).fit(X)
    out = clf.is_outlier(X)
    assert not out[:26].any()
    assert out[26:].all()


def test_majority_vote_per_track():
    votes = majority_vote([1, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 1, 0, 1])
    assert votes == {1: 0, 2: 1, 3: 0}  # égalité : la plus petite étiquette


def test_majority_vote_ignores_unknown():
    assert majority_vote([5, 5, 5], [UNKNOWN_TEAM, UNKNOWN_TEAM, 1]) == {5: 1}


def test_goalkeeper_joins_nearest_team():
    out = assign_by_centroid({9: np.array([5.0, 34.0]), 10: np.array([100.0, 30.0])},
                             {0: np.array([30.0, 34.0]), 1: np.array([75.0, 34.0])})
    assert out == {9: 0, 10: 1}


def test_grass_is_masked():
    cfg = TeamsConfig()
    crop = np.zeros((20, 20, 3), np.uint8)
    crop[:] = (50, 140, 40)  # pelouse
    crop[5:15, 5:15] = (40, 40, 200)  # maillot rouge
    color = dominant_color_lab(crop, cfg)
    np.testing.assert_allclose(color, _lab((40, 40, 200)), atol=2)


def test_jersey_color_from_box():
    cfg = TeamsConfig()
    frame = np.zeros((200, 200, 3), np.uint8)
    frame[:] = (50, 140, 40)
    frame[50:150, 80:120] = (240, 240, 240)
    color = jersey_color(frame, np.array([80, 50, 120, 150]), cfg)
    assert color is not None and color[0] > 200
