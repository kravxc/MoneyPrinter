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
                "ffmpeg", "-v", "error", "-hide_banner", "-y",
                "-ss", f"{sec:.3f}",
                "-i", str(input_path),
                "-frames:v", "1",
                "-vf", "scale=640:-2",
                "-q:v", "4",
                out,
            ],
            stderr=subprocess.DEVNULL,
        )
        return (sec, out)

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        frames = list(
            tqdm(pool.map(_extract, enumerate(times)), total=len(times),
                 desc="Баннеры (кадры)", unit="кадр")
        )
    return frames


def ocr_result(path: str) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """Распознаёт текст на кадре.

    Возвращает [(text, box)], где box = (x0, y0, x1, y1) в пикселях кадра.
    """
    try:
        result, _ = _get_engine()(str(path))
    except Exception:
        return []
    if not result:
        return []
    out: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for item in result:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        text = item[1] if isinstance(item[1], str) else ""
        box = item[0]
        xs = ys = None
        try:
            if isinstance(box, (list, tuple)) and box:
                points = [p for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                if points:
                    xs = [float(p[0]) for p in points]
                    ys = [float(p[1]) for p in points]
                else:
                    flat = [float(v) for v in box]
                    xs, ys = flat[0::2], flat[1::2]
        except (TypeError, ValueError):
            xs = ys = None
        if text and xs and ys:
            out.append((text, (min(xs), min(ys), max(xs), max(ys))))
    return out


def ocr_text(path: str) -> str:
    """Распознаёт текст на кадре, возвращает lowercase-строку."""
    return " ".join(t for t, _ in ocr_result(path)).lower()


def is_banner_text(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in BANNER_KEYWORDS)


def _boxes_to_crop(
    boxes: List[Tuple[float, float, float, float]],
    img_w: float,
    img_h: float,
    margin: float = 0.04,
) -> dict:
    """По позициям банер-текста вычисляет, какой край и насколько кадрировать.

    Возвращает dict вида {"bottom": 0.15}. Значения от 0.05 до 0.4.
    Горизонтальный банер (верх/низ кадра) кадрируется только сверху/снизу;
    боковые края учитываются лишь если банер вертикальный.
    """
    if not boxes or img_w <= 0 or img_h <= 0:
        return {}
    h_edges: dict = {}
    v_edges: dict = {}
    for x0, y0, x1, y1 in boxes:
        w = (x1 - x0) / img_w
        yc = (y0 + y1) / 2 / img_h
        xc = (x0 + x1) / 2 / img_w
        if yc >= 0.5:
            h_edges["bottom"] = max(h_edges.get("bottom", 0.0), 1.0 - (y1 / img_h) + margin)
        else:
            h_edges["top"] = max(h_edges.get("top", 0.0), (y0 / img_h) + margin)
        if w < 0.6:
            if xc < 0.5:
                v_edges["left"] = max(v_edges.get("left", 0.0), x0 / img_w + margin)
            else:
                v_edges["right"] = max(v_edges.get("right", 0.0), 1.0 - (x1 / img_w) + margin)
    edges = dict(h_edges)
    if not h_edges:  # вертикальный банер — смотрим бока
        edges.update(v_edges)
    return {e: round(min(max(r, 0.05), 0.4), 3) for e, r in edges.items() if r >= 0.05}


def detect_banner_crop(
    input_path: str,
    duration: float,
    video_width: int,
    video_height: int,
    interval_sec: Optional[float] = None,
    jobs: int = 4,
) -> dict:
    """Определяет кадрирование, убирающее банер по размеру кадра.

    Хронометраж не меняется: банер выносится за кадр срезом края.
    Возвращает {"bottom": 0.15} или пустой dict, если банер не найден.
    """
    if interval_sec is None:
        interval_sec = choose_interval(duration)

    # кадры извлекаются шириной 640 — пересчитываем высоту
    img_w = 640.0
    img_h = float(int(640.0 * video_height / max(1, video_width) / 2) * 2)
    if img_h <= 0:
        img_h = 360.0

    banner_boxes: List[Tuple[float, float, float, float]] = []
    with tempfile.TemporaryDirectory(prefix="moneyprinter_banner_") as td:
        frames = sample_frames(input_path, interval_sec, td, duration, jobs)
        for t, path in tqdm(frames, desc="OCR баннеров", unit="кадр"):
            for text, box in ocr_result(path):
                if is_banner_text(text):
                    banner_boxes.append(box)

    if not banner_boxes:
        return {}
    crop = _boxes_to_crop(banner_boxes, img_w, img_h)
    return crop


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