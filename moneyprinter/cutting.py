"""Нарезка клипов и конвертация в вертикальный формат 9:16."""

from __future__ import annotations

from .media import run_with_progress
from .models import ClipCandidate

TARGET_W, TARGET_H = 1080, 1920


def _sec(ts: float) -> str:
    return f"{max(0.0, ts):.3f}"


def _progress_args(candidate: ClipCandidate, bar, offset: float):
    """Общие параметры для прогресс-бара нарезки."""
    return dict(total=candidate.duration, bar=bar, initial=offset)


def cut_clip(
    input_path: str,
    candidate: ClipCandidate,
    out_path: str,
    video_codec: str = "libx264",
    crf: int = 20,
    preset: str = "fast",
    bar=None,
    offset: float = 0.0,
) -> str:
    """Режет фрагмент [start, end] без изменения пропорций."""
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(input_path),
        "-ss", _sec(candidate.start),
        "-to", _sec(candidate.end),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", video_codec, "-preset", preset, "-crf", str(crf),
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    run_with_progress(cmd, **_progress_args(candidate, bar, offset))
    return out_path


def make_vertical(
    input_path: str,
    candidate: ClipCandidate,
    out_path: str,
    blur_bg: bool = True,
    width: int = TARGET_W,
    height: int = TARGET_H,
    crf: int = 20,
    preset: str = "fast",
    bar=None,
    offset: float = 0.0,
) -> str:
    """Конвертирует фрагмент в вертикальный 9:16.

    Режим blur_bg: заполняем фон размытой растянутой копией и накладываем
    исходник по центру. Иначе — кадрируем (crop) по центру.
    """
    if blur_bg:
        filter_complex = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,boxblur=20:5[bg];"
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"setsar=1[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v]"
        )
    else:
        filter_complex = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1[v]"
        )

    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(input_path),
        "-ss", _sec(candidate.start),
        "-to", _sec(candidate.end),
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    run_with_progress(cmd, **_progress_args(candidate, bar, offset))
    return out_path