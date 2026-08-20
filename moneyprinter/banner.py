"""Визуальный детектор рекламных баннеров (казино/беттинг) через OCR.

Сэмплируем кадры видео, распознаём текст через RapidOCR (локально,
бесплатно), ищем маркеры казино/беттинга и возвращаем временные интервалы,
где баннер виден. Сегменты, пересекающиеся с этими интервалами, помечаются
как рекламные и вырезаются из клипов.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

from tqdm import tqdm

from .media import run_with_progress
from .models import TimestampedText

_SHOWINFO_RE = re.compile(r"pts_time:(\d+(?:\.\d+)?)")

# Маркеры казино/беттинга для OCR-текста баннеров (lowercase-подстроки)
BANNER_KEYWORDS = [
    # русские
    "казино", "ставк", "букмекер", "тотализатор", "1xbet", "1хбет",
    "бетсити", "фонбет", "мелбет", "winline", "винлайн", "лига ставок",
    "депозит", "фриспин", "бонус", "игровой автомат", "слот", "джекпот",
    "вулкан", "pinnacle", "париматч", "золото казино",
    # английские
    "casino", "betting", "bet365", "bookmaker", "free spin", "jackpot",
    "wager", "slots", "bet", "deposit",
]

# Максимальное число кадров для OCR-анализа (интервал подбирается под него)
MAX_OCR_FRAMES = 300
DEFAULT_INTERVAL = 10.0


def choose_interval(duration: float) -> float:
    """Интервал сэмплирования: не реже DEFAULT_INTERVAL, но не больше MAX_OCR_FRAMES."""
    return max(DEFAULT_INTERVAL, duration / MAX_OCR_FRAMES)


def ensure_ocr(auto_install: bool = True) -> bool:
    """Проверяет наличие RapidOCR; при auto_install сам ставит его."""
    try:
        import rapidocr_onnxruntime  # noqa: F401

        return True
    except ImportError:
        pass
    if not auto_install:
        return False
    print("[i] RapidOCR не найден — устанавливаю (для детекции рекламных баннеров)...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "rapidocr_onnxruntime"]
        )
        import rapidocr_onnxruntime  # noqa: F401

        return True
    except Exception:
        print("[warn] Не удалось установить RapidOCR — детекция баннеров отключена")
        return False


_ocr_engine = None


def _get_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
    return _ocr_engine


def sample_frames(
    input_path: str, interval_sec: float, out_dir: str, duration: float
) -> List[Tuple[float, str]]:
    """Извлекает кадр каждые interval_sec; возвращает [(time, path)]."""
    stderr = run_with_progress(
        [
            "ffmpeg", "-v", "info", "-hwaccel", "auto",
            "-i", str(input_path),
            "-vf", f"fps=1/{interval_sec},showinfo",
            "-q:v", "4",
            os.path.join(out_dir, "frame_%06d.jpg"),
        ],
        total=duration,
        desc="Баннеры (кадры)",
    )
    times: List[float] = []
    for line in stderr.splitlines():
        m = _SHOWINFO_RE.search(line)
        if m and "n:" in line:
            times.append(float(m.group(1)))
    paths = [os.path.join(out_dir, f"frame_{i + 1:06d}.jpg") for i in range(len(times))]
    return list(zip(times, paths))


def ocr_text(path: str) -> str:
    """Распознаёт текст на кадре, возвращает lowercase-строку."""
    try:
        result, _ = _get_engine()(str(path))
    except Exception:
        return ""
    if not result:
        return ""
    texts: List[str] = []
    for item in result:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], str):
            texts.append(item[1])
    return " ".join(texts).lower()


def is_banner_text(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in BANNER_KEYWORDS)


def detect_banner_ranges(
    input_path: str,
    duration: float,
    interval_sec: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Возвращает интервалы времени, где на экране рекламный баннер."""
    if interval_sec is None:
        interval_sec = choose_interval(duration)

    with tempfile.TemporaryDirectory(prefix="moneyprinter_banner_") as td:
        frames = sample_frames(input_path, interval_sec, td, duration)
        hits: List[float] = []
        for t, path in tqdm(frames, desc="OCR баннеров", unit="кадр"):
            if is_banner_text(ocr_text(path)):
                hits.append(t)

    if not hits:
        return []
    hits.sort()
    ranges: List[Tuple[float, float]] = []
    start = prev = hits[0]
    for t in hits[1:]:
        if t - prev > interval_sec * 2:
            ranges.append((start, prev + interval_sec))
            start = t
        prev = t
    ranges.append((start, prev + interval_sec))
    return ranges


def mark_segments_by_ranges(
    segments: List[TimestampedText],
    ranges: List[Tuple[float, float]],
    tolerance: float = 1.0,
) -> List[TimestampedText]:
    """Помечает сегменты, пересекающиеся с интервалами баннеров."""
    for seg in segments:
        for r0, r1 in ranges:
            if seg.end >= r0 - tolerance and seg.start <= r1 + tolerance:
                seg.is_ad = True
                break
    return segments