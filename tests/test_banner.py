import numpy as np

from moneyprinter.banner import (
    _boxes_to_crop,
    _expand_and_merge,
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
    assert choose_interval(3600.0) >= 3600.0 / 900
    assert choose_interval(3600.0) <= 6.0
    assert choose_interval(60.0) == 5.0


def test_expand_and_merge_covers_gaps_between_hits():
    # хиты на 100 и 130 с шагом 10 → расширение ±15 и слияние в один интервал
    ranges = _expand_and_merge([100.0, 130.0], interval_sec=10.0, duration=1000.0)
    assert len(ranges) == 1
    assert ranges[0][0] == 85.0
    assert ranges[0][1] == 145.0


def test_expand_and_merge_separates_distant_hits():
    ranges = _expand_and_merge([10.0, 400.0], interval_sec=10.0, duration=1000.0)
    assert len(ranges) == 2


def test_expand_and_merge_clamps_to_video():
    ranges = _expand_and_merge([5.0], interval_sec=10.0, duration=100.0)
    assert ranges[0][0] == 0.0
    assert ranges[0][1] == 20.0


def test_expand_and_merge_empty():
    assert _expand_and_merge([], interval_sec=10.0, duration=100.0) == []


def test_boxes_to_crop_bottom():
    # банер в нижней части кадра 640x360 → кадрируем низ
    crop = _boxes_to_crop([(100.0, 300.0, 500.0, 350.0)], 640.0, 360.0)
    assert "bottom" in crop
    assert 0.05 <= crop["bottom"] <= 0.15


def test_boxes_to_crop_top():
    crop = _boxes_to_crop([(100.0, 10.0, 500.0, 40.0)], 640.0, 360.0)
    assert "top" in crop
    assert 0.05 <= crop["top"] <= 0.15


def test_boxes_to_crop_empty():
    assert _boxes_to_crop([], 640.0, 360.0) == {}
    assert _boxes_to_crop([(10.0, 10.0, 20.0, 20.0)], 0.0, 360.0) == {}


def test_mark_segments_by_ranges_no_ranges():
    segs = [_seg(0, 5)]
    mark_segments_by_ranges(segs, [])
    assert not segs[0].is_ad