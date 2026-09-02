"""Тесты генерации «нейрослоп»-клипов (slop)."""

from __future__ import annotations

from moneyprinter import slop


def test_random_story_generated():
    s1 = slop._random_abject_story(42)
    s2 = slop._random_abject_story(42)
    assert s1 == s2  # детерминировано по сиду
    assert s1.strip()
    assert len(s1) > 20


def test_random_story_different_seeds():
    s1 = slop._random_abject_story(1)
    s2 = slop._random_abject_story(2)
    assert s1 != s2


def test_split_scenes():
    s = "Я клубника. Ты клубника. Почему у нас банан? Банан улыбается!"
    scenes = slop.split_scenes(s)
    assert 4 <= len(scenes) <= 12
    assert all(sc.strip() for sc in scenes)


def test_split_scenes_limit():
    long_text = " ".join(f"Предложение {i}." for i in range(20))
    assert len(slop.split_scenes(long_text)) <= 12


def test_resolve_story_order():
    # приоритет: прямой текст > файл > случайная
    cfg = slop.SlopConfig(story="  Taleseeb  ", story_file="/nonexistent")
    assert slop.resolve_story(cfg) == "Taleseeb"


def test_detect_tts_returns_something_or_error():
    # на любой ОС должен быть хоть один движок или явный RuntimeError позже
    engine = slop._detect_tts()
    assert isinstance(engine, str)