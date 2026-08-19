"""Нарезка клипов и конвертация в вертикальный формат 9:16."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .media import FFmpegError, run_with_progress
from .models import ClipCandidate, TimestampedText

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


def _build_ass(segments: List[TimestampedText], width: int = TARGET_W, height: int = TARGET_H) -> str:
    """Собирает ASS-субтитры для вшивания в вертикальный клип."""
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,68,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,2,80,80,60,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for seg in segments:
        text = seg.text.strip().replace("\n", " ")
        if not text:
            continue
        text = text.replace("{", "\\{").replace("}", "\\}")
        start = _ass_time(seg.start)
        end = _ass_time(seg.end)
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
    return "\n".join(lines) + "\n"


def _ass_time(ts: float) -> str:
    ts = max(0.0, ts)
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    cs = int(round((ts % 1) * 100))
    if cs == 100:
        cs = 0
        s += 1
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def make_vertical(
    input_path: str,
    candidate: ClipCandidate,
    out_path: str,
    segments: Optional[List[TimestampedText]] = None,
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
    ratio = f"{width}:{height}"
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

    if segments:
        ass_path = str(Path(out_path).with_suffix(".ass"))
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(_build_ass(segments, width, height))
        filter_complex += f";[v]ass={ass_path}[v]"

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
    try:
        run_with_progress(cmd, **_progress_args(candidate, bar, offset))
    except FFmpegError:
        if segments:
            Path(ass_path).unlink(missing_ok=True)
        raise
    if segments:
        Path(ass_path).unlink(missing_ok=True)
    return out_path


def write_srt(segments: List[TimestampedText], out_path: str) -> str:
    """Пишет SRT-файл рядом с клипом (удобно для загрузки в соцсети)."""
    lines: List[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_time(seg.start)} --> {_srt_time(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def _srt_time(ts: float) -> str:
    ts = max(0.0, ts)
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    ms = int((ts % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"