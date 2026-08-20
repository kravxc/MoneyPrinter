import numpy as np

from moneyprinter.banner import (
    choose_interval,
    is_banner_text,
    mark_segments_by_ranges,
)
from moneyprinter.models import TimestampedText


def _seg(start, end, text="реплика"):
    return TimestampedText(start=start, end=end, text=text)


def test_is_banner_text_positive():
    assert is_banner_text("казино вулкан бонус")
    assert is_banner_text("1XBET фриспины")
    assert is_banner_text("CASINO BET 365")


def test_is_banner_text_negative():
    assert not is_banner_text("стрим по игре в шутер")
    assert not is_banner_text("обычная речь без рекламы")


def test_mark_segments_by_ranges():
    segs = [_seg(0, 5), _seg(10, 15), _seg(20, 25)]
    mark_segments_by_ranges(segs, [(9.0, 16.0)], tolerance=1.0)
    assert not segs[0].is_ad
    assert segs[1].is_ad
    assert not segs[2].is_ad


def test_mark_segments_by_ranges_tolerance():
    segs = [_seg(0, 5), _seg(10, 15)]
    mark_segments_by_ranges(segs, [(4.5, 9.5)], tolerance=1.0)
    assert segs[0].is_ad  # конец 5 >= 4.5-1
    assert segs[1].is_ad  # старт 10 <= 9.5+1 (допуск в 1с)


def test_choose_interval_caps_frames():
    # очень длинное видео → интервал растёт, чтобы кадров было не больше лимита
    assert choose_interval(3600.0) >= 3600.0 / 300
    assert choose_interval(60.0) == 10.0
    assert choose_interval(60.0) <= 10.0 + 1e-9


def test_mark_segments_by_ranges_no_ranges():
    segs = [_seg(0, 5)]
    mark_segments_by_ranges(segs, [])
    assert not segs[0].is_ad