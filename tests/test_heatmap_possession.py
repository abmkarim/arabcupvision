import numpy as np
import pandas as pd

from arabcupvision.analytics.heatmap import heatmap_grid, render_heatmap
from arabcupvision.analytics.possession import possession_per_frame, possession_share
from arabcupvision.pitch.geometry import PitchDimensions


def test_heatmap_peak_at_player_location():
    pts = np.tile([[26.25, 17.0]], (50, 1))
    grid = heatmap_grid(pts, 105, 68, bins=(42, 27), sigma=1.0)
    assert grid.shape == (27, 42)
    assert grid.max() == 1.0
    r, c = np.unravel_index(np.argmax(grid), grid.shape)
    assert abs(c - 10) <= 1 and abs(r - 6) <= 1


def test_heatmap_ignores_out_of_pitch_and_nan():
    grid = heatmap_grid(np.array([[np.nan, 1.0], [200.0, 10.0]]), 105, 68)
    assert grid.sum() == 0


def test_render_heatmap_png(tmp_path):
    grid = heatmap_grid(np.random.default_rng(0).uniform([0, 0], [105, 68], (200, 2)), 105, 68)
    out = tmp_path / "h.png"
    render_heatmap(grid, PitchDimensions(), "test", out)
    assert out.exists() and out.stat().st_size > 1000


def test_possession_nearest_player_with_threshold():
    players = pd.DataFrame({
        "frame": [0, 0, 1, 1], "track_id": [1, 2, 1, 2], "team": [0, 1, 0, 1],
        "pitch_x": [10.0, 20.0, 10.0, 20.0], "pitch_y": [10.0, 10.0, 10.0, 10.0],
    })
    ball = pd.DataFrame({"frame": [0, 1], "pitch_x": [19.0, 50.0], "pitch_y": [10.0, 10.0]})
    per = possession_per_frame(players, ball, max_distance=3.0)
    assert per["team"].tolist() == [1, -1]
    assert per.loc[0, "track_id"] == 2
    assert possession_share(per) == {"team_0": 0.0, "team_1": 100.0}
