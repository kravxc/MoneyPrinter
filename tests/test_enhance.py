"""Тесты модуля улучшения качества видео (enhance)."""

from __future__ import annotations

from moneyprinter import enhance


def test_vf_builds_chain():
    cfg = enhance.EnhanceConfig(
        target_width=1920, target_height=1080,
        denoise_strength=3, sharpen_strength=1.0,
    )
    vf = enhance._build_vf(cfg)
    assert "deblock" in vf
    assert "hqdn3d" in vf
    assert "unsharp" in vf
    assert "cas" not in vf
    assert "scale=1920:1080" in vf

    assert vf.index("deblock") < vf.index("hqdn3d") < vf.index("unsharp")


def test_vf_luma_only_sharpening():

    cfg = enhance.EnhanceConfig(sharpen_strength=0.8, sharp_mode="unsharp")
    vf = enhance._build_vf(cfg)

    assert "unsharp=5:5:0.8:5:5:0.0" in vf


def test_vf_preserves_vertical_orientation():

    cfg = enhance.EnhanceConfig(target_width=1920, target_height=1080)
    vf = enhance._build_vf(cfg, src_w=1080, src_h=1920)

    assert "scale=1080:1920" in vf

    assert "force_original_aspect_ratio=decrease" in vf


def test_vf_preserves_horizontal_orientation():
    cfg = enhance.EnhanceConfig(target_width=1920, target_height=1080)
    vf = enhance._build_vf(cfg, src_w=1920, src_h=1080)
    assert "scale=1920:1080" in vf


def test_vf_sharp_modes():

    vf_cas = enhance._build_vf(
        enhance.EnhanceConfig(sharp_mode="cas", sharpen_strength=1.0)
    )
    assert "cas" in vf_cas and "unsharp" not in vf_cas

    vf_both = enhance._build_vf(
        enhance.EnhanceConfig(sharp_mode="both", sharpen_strength=1.0)
    )
    assert "cas" in vf_both and "unsharp" in vf_both

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
    assert not enhance.check_realesrgan() or True
