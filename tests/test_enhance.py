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
    assert "cas" in vf  # дефолтный режим cas+unsharp
    assert "scale=1920:1080" in vf
    # шумоподавление идёт ДО резкости
    assert vf.index("hqdn3d") < vf.index("unsharp")


def test_vf_preserves_vertical_orientation():
    # вертикальное видео (1080x1920) не должно сплющиваться в горизонтальный
    cfg = enhance.EnhanceConfig(target_width=1920, target_height=1080)
    vf = enhance._build_vf(cfg, src_w=1080, src_h=1920)
    # цель переставляется: 1080x1920 (вертикальное)
    assert "scale=1080:1920" in vf
    # содержимое не искажается
    assert "force_original_aspect_ratio=decrease" in vf


def test_vf_preserves_horizontal_orientation():
    cfg = enhance.EnhanceConfig(target_width=1920, target_height=1080)
    vf = enhance._build_vf(cfg, src_w=1920, src_h=1080)
    assert "scale=1920:1080" in vf


def test_vf_sharp_modes():
    # cas-only
    vf_cas = enhance._build_vf(
        enhance.EnhanceConfig(sharp_mode="cas", sharpen_strength=1.0)
    )
    assert "cas" in vf_cas and "unsharp" not in vf_cas
    # unsharp-only
    vf_un = enhance._build_vf(
        enhance.EnhanceConfig(sharp_mode="unsharp", sharpen_strength=1.0)
    )
    assert "unsharp" in vf_un and "cas" not in vf_un
    # off
    vf_off = enhance._build_vf(
        enhance.EnhanceConfig(sharp_mode="off", sharpen_strength=1.0)
    )
    assert "cas" not in vf_off and "unsharp" not in vf_off


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
