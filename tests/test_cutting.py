import shutil
import subprocess

import pytest

from moneyprinter.cutting import _ass_time, _build_ass, cut_clip, make_vertical
from moneyprinter.media import probe
from moneyprinter.models import ClipCandidate, TimestampedText

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("videos") / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=6:size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )
    return str(path)


def test_probe(sample_video):
    info = probe(sample_video)
    assert info.width == 640 and info.height == 360
    assert info.has_audio
    assert info.duration > 5


def test_cut_clip(sample_video, tmp_path):
    out = str(tmp_path / "cut.mp4")
    cand = ClipCandidate(start=1.0, end=3.0)
    cut_clip(sample_video, cand, out)
    info = probe(out)
    assert 1.5 < info.duration < 2.5
    assert info.width == 640


def test_make_vertical_blur(sample_video, tmp_path):
    out = str(tmp_path / "vert.mp4")
    cand = ClipCandidate(start=1.0, end=3.0)
    make_vertical(sample_video, cand, out, blur_bg=True)
    info = probe(out)
    assert info.width == 1080 and info.height == 1920


def test_make_vertical_crop(sample_video, tmp_path):
    out = str(tmp_path / "vert_crop.mp4")
    cand = ClipCandidate(start=1.0, end=3.0)
    make_vertical(sample_video, cand, out, blur_bg=False)
    info = probe(out)
    assert info.width == 1080 and info.height == 1920


def test_ass_time():
    assert _ass_time(0) == "0:00:00.00"
    assert _ass_time(75.5) == "0:01:15.50"
    assert _ass_time(3661.1) == "1:01:01.10"


def test_build_ass():
    segs = [TimestampedText(start=1.0, end=2.5, text="привет мир")]
    ass = _build_ass(segs)
    assert "[Events]" in ass
    assert "Dialogue:" in ass
    assert "привет мир" in ass