"""Нарезка клипов и конвертация в вертикальный формат 9:16."""

from __future__ import annotations

from .media import run_with_progress
from .models import ClipCandidate

TARGET_W, TARGET_H = 1080, 1920


def _sec(ts: float) -> str:
    return f"{max(0.0, ts):.3f}"


def _crop_filters(crop_edges: Optional[dict]) -> str:
    """Строит цепочку crop-фильтров, убирающих банер с края кадра."""
    if not crop_edges:
        return ""
    parts = []
    for edge, ratio in crop_edges.items():
        r = max(0.0, min(0.5, ratio))
        if edge == "bottom":
            parts.append(f"crop=iw:ih*{1 - r:.4f}:0:0")
        elif edge == "top":
            parts.append(f"crop=iw:ih*{1 - r:.4f}:0:ih*{r:.4f}")
        elif edge == "left":
            parts.append(f"crop=iw*{1 - r:.4f}:ih:iw*{r:.4f}:0")
        elif edge == "right":
            parts.append(f"crop=iw*{1 - r:.4f}:ih:0:0")
    return ",".join(parts)


def _progress_args(candidate: ClipCandidate, bar, offset: float):
    """Общие параметры для прогресс-бара нарезки."""
    return dict(total=candidate.duration, bar=bar, initial=offset)


def cut_clip(
    input_path: str,
    candidate: ClipCandidate,
    out_path: str,
    video_codec: str = "libx264",
    crf: int = 20,
    preset: str = "veryfast",
    bar=None,
    offset: float = 0.0,
    crop_edges: Optional[dict] = None,
) -> str:
    """Режет фрагмент [start, end] без изменения пропорций.

    crop_edges — dict {"bottom": 0.15}: срезает банер с края кадра,
    хронометраж клипа не меняется.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-hwaccel", "auto",
        "-i", str(input_path),
        "-ss", _sec(candidate.start),
        "-to", _sec(candidate.end),
        "-map", "0:v:0", "-map", "0:a:0?",
    ]
    crops = _crop_filters(crop_edges)
    if crops:
        cmd += ["-vf", crops]
    cmd += [
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
    preset: str = "veryfast",
    bar=None,
    offset: float = 0.0,
    crop_edges: Optional[dict] = None,
) -> str:
    """Конвертирует фрагмент в вертикальный 9:16.

    Режим blur_bg: заполняем фон размытой растянутой копией и накладываем
    исходник по центру. Иначе — кадрируем (crop) по центру.
    crop_edges — dict {"bottom": 0.15}: срезает банер с края кадра ДО
    конвертации; хронометраж клипа не меняется.
    """
    crops = _crop_filters(crop_edges)
    pre = f"[0:v]{crops}[c];" if crops else ""
    src = "[c]" if crops else "[0:v]"

    if blur_bg:
        filter_complex = (
            pre
            + f"{src}scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,boxblur=20:5[bg];"
            f"{src}scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"setsar=1[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v]"
        )
    else:
        filter_complex = (
            pre
            + f"{src}scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1[v]"
        )

    cmd = [
        "ffmpeg", "-v", "error", "-y", "-hwaccel", "auto",
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