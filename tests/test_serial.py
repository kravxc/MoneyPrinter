"""Тесты режима 'сериал → микро-серии'."""

from __future__ import annotations

from moneyprinter import serial


def test_build_parts_even_split():
    cfg = serial.SerialConfig(input_path="x.mp4", part_duration=60.0)
    parts = serial._build_parts(cfg, duration=185.0)
    assert len(parts) == 4  # 60,60,60,5
    assert parts[0] == (1, 0.0, 60.0)
    assert parts[1] == (2, 60.0, 120.0)
    assert parts[3] == (4, 180.0, 185.0)


def test_build_parts_with_start_end_and_max():
    cfg = serial.SerialConfig(input_path="x.mp4", part_duration=30.0, start=10.0, end=100.0, max_parts=2)
    parts = serial._build_parts(cfg, duration=1000.0)
    assert len(parts) == 2
    assert parts[0] == (1, 10.0, 40.0)
    assert parts[1] == (2, 40.0, 70.0)


def test_build_parts_shorter_than_part():
    cfg = serial.SerialConfig(input_path="x.mp4", part_duration=120.0)
    parts = serial._build_parts(cfg, duration=45.0)
    assert len(parts) == 1
    assert parts[0] == (1, 0.0, 45.0)


def test_caption_includes_part_numbers():
    cfg = serial.SerialConfig(input_path="x.mp4", series_title="Уроки химии", episode=1)
    caption, tags = serial._make_caption_and_tags(cfg, part_idx=3, total=12, duration=60.0)
    assert "Часть 3" in caption
    assert "Серия 1" in caption
    assert "/12" not in caption  # формат "Часть N", без N/Y
    tags_str = " ".join(f"#{t}" for t in tags)
    assert "химии" in tags_str  # тематический тег по названию сериала


def test_caption_with_custom_base_tags():
    cfg = serial.SerialConfig(input_path="x.mp4", series_title="X", episode=2, base_hashtags=["customtag"])
    _, tags = serial._make_caption_and_tags(cfg, 1, 5, 60.0)
    assert "customtag" in tags
