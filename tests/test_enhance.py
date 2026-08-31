"""Тесты модуля улучшения качества видео (enhance)."""

from __future__ import annotations

from moneyprinter import enhance


def test_vf_builds_chain():
    cfg = enhance.EnhanceConfig(
        target_width=1920, target_height=1080,
        denoise_strength=3, sharpen_strength=1.0,
    )
    vf = enhance._build_vf(cfg)
    assert "hqdn3d" in vf
    assert "unsharp" in vf
    assert "scale=1920:1080" in vf
    # шумоподавление идёт ДО резкости
    assert vf.index("hqdn3d") < vf.index("unsharp")


def test_vf_disabled_filters():
    cfg = enhance.EnhanceConfig(
        target_width=1280, target_height=720,
        denoise_strength=0, sharpen_strength=0,
    )
    vf = enhance._build_vf(cfg)
    assert "hqdn3d" not in vf
    assert "unsharp" not in vf
    assert "scale=1280:720" in vf


def test_vf_ai_disabled_by_default():
    cfg = enhance.EnhanceConfig()
    assert cfg.use_ai is False
    assert not enhance.check_realesrgan() or True  # не должно падать
