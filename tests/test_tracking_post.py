import numpy as np
import pandas as pd

from arabcupvision.tracking.postprocess import interpolate_ball, smooth_tracks


def test_ball_interpolation_fills_short_gaps():
    ball = pd.DataFrame({"frame": [0, 1, 4, 5], "x": [0.0, 10.0, 40.0, 50.0],
                         "y": [0.0, 0.0, 0.0, 0.0]})
    out = interpolate_ball(ball, range(6), max_gap=3).set_index("frame")
    np.testing.assert_allclose(out.loc[[2, 3], "x"], [20.0, 30.0])
    assert out.loc[[2, 3], "interpolated"].all()
    assert not out.loc[[0, 1, 4, 5], "interpolated"].any()


def test_ball_interpolation_respects_max_gap():
    ball = pd.DataFrame({"frame": [0, 10], "x": [0.0, 100.0], "y": [0.0, 0.0]})
    out = interpolate_ball(ball, range(11), max_gap=3).set_index("frame")
    assert out.loc[1:9, "x"].isna().all()


def test_ball_interpolation_no_extrapolation_and_skip():
    ball = pd.DataFrame({"frame": [4, 8], "x": [4.0, 8.0], "y": [0.0, 0.0]})
    out = interpolate_ball(ball, [0, 2, 4, 6, 8, 10], max_gap=2).set_index("frame")
    assert np.isnan(out.loc[0, "x"]) and np.isnan(out.loc[10, "x"])
    assert out.loc[6, "x"] == 6.0


def test_ball_interpolation_empty():
    out = interpolate_ball(pd.DataFrame(columns=["frame", "x", "y"]), [0, 1], 5)
    assert len(out) == 2 and out["x"].isna().all()


def test_smoothing_per_track():
    df = pd.DataFrame({"track_id": [1] * 5 + [2] * 5, "frame": list(range(5)) * 2,
                       "x": [0, 10, 0, 10, 0] + [100] * 5})
    out = smooth_tracks(df, ["x"], window=3)
    t1 = out[out.track_id == 1]["x"].to_numpy()
    assert t1.max() < 10 and t1.min() > 0
    assert (out[out.track_id == 2]["x"] == 100).all()
