import json

import numpy as np

from arabcupvision.pitch.geometry import PitchDimensions, draw_pitch
from arabcupvision.pitch.homography import (
    HomographyTracker,
    approximate_homography,
    estimate_homography,
    load_calibration,
    transform_points,
)


def _true_h() -> np.ndarray:
    return np.array([[0.08, 0.01, -5.0], [0.002, 0.12, -10.0], [0.0001, 0.0006, 1.0]])


def test_homography_recovers_known_points():
    H_true = _true_h()
    img = np.array([[100, 120], [900, 110], [1200, 650], [50, 700], [640, 360], [300, 500]],
                   dtype=float)
    pitch = transform_points(H_true, img)
    H = estimate_homography(img, pitch)
    assert H is not None
    np.testing.assert_allclose(transform_points(H, img), pitch, atol=1e-6)
    probe = np.array([[500.0, 400.0]])
    np.testing.assert_allclose(transform_points(H, probe), transform_points(H_true, probe),
                               atol=1e-6)


def test_homography_needs_four_points():
    assert estimate_homography(np.zeros((3, 2)), np.zeros((3, 2))) is None


def test_approximate_homography_maps_corners():
    H = approximate_homography(1280, 720, 105, 68)
    out = transform_points(H, np.array([[0, 0], [1280, 720]]))
    np.testing.assert_allclose(out, [[0, 0], [105, 68]], atol=1e-6)


def test_calibration_file_with_scale(tmp_path):
    H_true = _true_h()
    img = np.array([[100, 120], [900, 110], [1200, 650], [50, 700]], dtype=float)
    path = tmp_path / "calib.json"
    path.write_text(json.dumps({"image_points": img.tolist(),
                                "pitch_points": transform_points(H_true, img).tolist()}))
    H = load_calibration(path, scale=0.5)
    np.testing.assert_allclose(transform_points(H, img * 0.5), transform_points(H_true, img),
                               atol=1e-6)


def test_tracker_keeps_last_valid():
    tr = HomographyTracker(recompute_interval=2)
    assert tr.should_update()
    tr.update(np.eye(3), "keypoints")
    tr.update(None, "keypoints")
    np.testing.assert_array_equal(tr.current, np.eye(3))
    assert not tr.should_update()
    assert tr.should_update()


def test_pitch_landmarks_and_drawing():
    dims = PitchDimensions()
    lm = dims.landmarks()
    assert lm["corner_br"] == (105.0, 68.0)
    np.testing.assert_allclose(lm["left_box_top"], (16.5, 13.84))
    img, scale = draw_pitch(dims, width_px=525, margin_px=10)
    assert img.shape[1] == 525 and abs(scale - 505 / 105) < 1e-9
