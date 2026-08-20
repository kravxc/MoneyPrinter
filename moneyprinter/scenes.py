"""Детекция смены сцен.

Сначала пробуем PySceneDetect (точнее), иначе падаем на встроенный
scene-фильтр ffmpeg — бесплатно и без лишних зависимостей.
"""

from __future__ import annotations

import re
from typing import List

from .media import run_with_progress
from .models import SceneBreak

_SHOWINFO_RE = re.compile(r"pts_time:(\d+(?:\.\d+)?)")


def _detect_ffmpeg(path: str, duration: float, threshold: float = 0.35) -> List[SceneBreak]:
    """Scene detection через ffmpeg: анализирует stderr-вывод showinfo."""
    stderr = run_with_progress(
        [
            "ffmpeg", "-v", "info", "-hwaccel", "auto",
            "-i", str(path),
            "-filter:v", f"select='gt(scene,{threshold})',showinfo",
            "-f", "null", "-",
        ],
        total=duration,
        desc="Сцены",
    )
    breaks: List[SceneBreak] = []
    for line in stderr.splitlines():
        m = _SHOWINFO_RE.search(line)
        if m and "n:" in line:
            t = float(m.group(1))
            if t > 1.0:  # игнорируем первый кадр
                breaks.append(SceneBreak(time=t))
    return breaks


def _detect_pyscenedetect(path: str, threshold: float = 27.0) -> List[SceneBreak]:
    try:
        from scenedetect import ContentDetector, detect
    except ImportError:
        raise ImportError("scenedetect не установлен")
    scenes = detect(str(path), ContentDetector(threshold=threshold))
    breaks: List[SceneBreak] = []
    for scene in scenes:
        start = scene[0].get_seconds()
        if start > 1.0:
            breaks.append(SceneBreak(time=start))
    return breaks


def detect_scenes(
    path: str,
    duration: float,
    threshold: float = 27.0,
    prefer_pyscenedetect: bool = True,
) -> List[SceneBreak]:
    """Определяет моменты смены сцен. Возвращает отсортированный список."""
    if prefer_pyscenedetect:
        try:
            breaks = _detect_pyscenedetect(path, threshold)
            if breaks:
                return breaks
        except ImportError:
            pass
        except Exception:
            pass
    return _detect_ffmpeg(path, duration, threshold / 60.0)