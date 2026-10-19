import numpy as np
import pandas as pd
import pytest

from arabcupvision.analytics.kinematics import player_statistics, track_kinematics


def test_constant_speed():
    t = np.arange(0, 10.04, 0.04)
    xy = np.stack([5.0 * t, np.zeros_like(t)], axis=1)  # 5 m/s = 18 km/h
    k = track_kinematics(t, xy, max_speed_kmh=40)
    assert k["distance_m"] == pytest.approx(5.0 * t[-1], rel=1e-6)
    assert k["max_speed_kmh"] == pytest.approx(18.0, rel=1e-6)
    assert np.nanmean(k["speed_kmh"]) == pytest.approx(18.0, rel=1e-6)


def test_outlier_jump_is_ignored():
    t = np.arange(0, 2.0, 0.1)
    x = 0.5 * np.arange(len(t))  # 5 m/s
    x[10:] += 30.0  # saut de 30 m en 0,1 s (erreur de suivi)
    xy = np.stack([x, np.zeros_like(x)], axis=1)
    k = track_kinematics(t, xy, max_speed_kmh=40)
    assert k["max_speed_kmh"] < 40
    assert k["distance_m"] == pytest.approx(0.5 * (len(t) - 2), rel=1e-6)


def test_single_point():
    k = track_kinematics(np.array([0.0]), np.array([[1.0, 1.0]]), 40)
    assert k["distance_m"] == 0.0 and k["max_speed_kmh"] == 0.0


def test_player_statistics_table():
    t = np.arange(0, 4, 0.5)
    rows = []
    for tid, speed in ((1, 2.0), (2, 4.0)):
        for ti in t:
            rows.append({"track_id": tid, "time_s": ti, "pitch_x": speed * ti, "pitch_y": 10.0,
                         "team": tid - 1, "role": "player"})
    stats, speeds = player_statistics(pd.DataFrame(rows), max_speed_kmh=40, window=1)
    assert list(stats["track_id"]) == [1, 2]
    assert stats.loc[1, "distance_m"] == pytest.approx(4.0 * 3.5)
    assert stats.loc[0, "max_speed_kmh"] == pytest.approx(7.2)
    assert stats.loc[0, "presence_s"] == pytest.approx(len(t) * 0.5)
    assert speeds.notna().sum() == 2 * (len(t) - 1)
