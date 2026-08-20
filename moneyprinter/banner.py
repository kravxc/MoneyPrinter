"""Визуальный детектор рекламных баннеров (казино/беттинг) через OCR.

Сэмплируем кадры видео, распознаём текст через RapidOCR (локально,
бесплатно), ищем маркеры казино/беттинга и возвращаем временные интервалы,
где баннер виден. Сегменты, пересекающиеся с этими интервалами, помечаются
как рекламные и вырезаются из клипов.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

from tqdm import tqdm

from .models import TimestampedText

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
MAX_OCR_FRAMES = 900
DEFAULT_INTERVAL = 5.0


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
    input_path: str,
    interval_sec: float,
    out_dir: str,
    duration: float,
    jobs: int = 4,
) -> List[Tuple[float, str]]:
    """Извлекает кадр каждые interval_sec через быстрый seek (-ss).

    Возвращает [(time, path)]. Каждый кадр — отдельный лёгкий вызов ffmpeg
    (декодируется 1 кадр), выполняется параллельно на нескольких ядрах,
    поэтому длинные ролики сэмплируются за минуты, а не за часы.
    """
    from concurrent.futures import ThreadPoolExecutor

    times: List[float] = []
    t = 0.0
    while t < duration - 1e-6:
        times.append(t)
        t += interval_sec
    if not times:
        times = [0.0]

    def _extract(args):
        idx, sec = args
        out = os.path.join(out_dir, f"frame_{idx + 1:06d}.jpg")
        subprocess.check_call(
            [
                "ffmpeg", "-v", "error", "-y",
                "-ss", f"{sec:.3f}",
                "-i", str(input_path),
                "-frames:v", "1",
                "-vf", "scale=640:-2",
                "-q:v", "4",
                out,
            ]
        )
        return (sec, out)

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        frames = list(
            tqdm(pool.map(_extract, enumerate(times)), total=len(times),
                 desc="Баннеры (кадры)", unit="кадр")
        )
    return frames


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


def _expand_and_merge(
    hits: List[float], interval_sec: float, duration: float
) -> List[Tuple[float, float]]:
    """Расширяет каждый хит на ±interval*1.5 и сливает соседние окна."""
    exp = interval_sec * 1.5
    windows = [(max(0.0, t - exp), min(duration, t + exp)) for t in hits]
    windows.sort()
    ranges: List[Tuple[float, float]] = []
    if not windows:
        return ranges
    cur_start, cur_end = windows[0]
    for s, e in windows[1:]:
        if s <= cur_end + interval_sec:
            cur_end = max(cur_end, e)
        else:
            ranges.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    ranges.append((cur_start, cur_end))
    return ranges


def detect_banner_ranges(
    input_path: str,
    duration: float,
    interval_sec: Optional[float] = None,
    jobs: int = 4,
) -> List[Tuple[float, float]]:
    """Возвращает интервалы времени, где на экране рекламный банер.

    Каждый подтверждённый кадр расширяется на ±interval*1.5 (чтобы закрыть
    промежутки между сэмплами и пропущенные OCR кадры), затем окна сливаются.
    Границы консервативнее, чем сам хит: лучше вырезать чуть лишнего,
    чем оставить банер в клипе.
    """
    if interval_sec is None:
        interval_sec = choose_interval(duration)

    with tempfile.TemporaryDirectory(prefix="moneyprinter_banner_") as td:
        frames = sample_frames(input_path, interval_sec, td, duration, jobs)
        hits: List[float] = []
        for t, path in tqdm(frames, desc="OCR баннеров", unit="кадр"):
            if is_banner_text(ocr_text(path)):
                hits.append(t)

    return _expand_and_merge(hits, interval_sec, duration)


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