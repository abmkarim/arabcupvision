import numpy as np
import pytest

from arabcupvision.config import RenderConfig
from arabcupvision.pitch.geometry import PitchDimensions
from arabcupvision.rendering.annotator import FrameItem, MiniMap, draw_ball, draw_player
from arabcupvision.video import VideoError, VideoWriter, iter_frames, probe, working_size


def _write(path, n=12, size=(160, 96), fps=10.0):
    with VideoWriter(path, fps, size, h264=False) as w:
        for i in range(n):
            frame = np.zeros((size[1], size[0], 3), np.uint8)
            frame[:, : 10 * (i + 1)] = 200
            w.write(frame)


def test_write_then_read(tmp_path):
    path = tmp_path / "clip.mp4"
    _write(path)
    info = probe(path)
    assert (info.width, info.height) == (160, 96)
    assert info.frame_count == 12 and info.fps == pytest.approx(10.0)
    frames = list(iter_frames(path))
    assert [i for i, _ in frames] == list(range(12))


def test_frame_skip_duration_and_resize(tmp_path):
    path = tmp_path / "clip.mp4"
    _write(path)
    got = list(iter_frames(path, frame_skip=3, max_seconds=1.0, size=(80, 48)))
    assert [i for i, _ in got] == [0, 3, 6, 9]
    assert got[0][1].shape == (48, 80, 3)
    w, h, scale = working_size(probe(path), 80)
    assert (w, h, scale) == (80, 48, 0.5)


def test_unreadable_video(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(VideoError):
        probe(bad)
    with pytest.raises(VideoError):
        probe(tmp_path / "missing.mp4")


def test_annotation_and_minimap():
    frame = np.zeros((360, 640, 3), np.uint8)
    item = FrameItem(np.array([100, 100, 130, 180]), 7, (0, 0, 255), (52.5, 34.0))
    draw_player(frame, item)
    draw_ball(frame, (300, 200), (255, 255, 255))
    before = frame.copy()
    mm = MiniMap(PitchDimensions(), RenderConfig(minimap_width=200))
    mm.overlay(frame, mm.render([item], (50.0, 30.0)))
    assert before[160:200, 90:140].any()
    assert not np.array_equal(before, frame)
