"""Генерация кандидатов и их скоринг.

Гибрид бесплатных эвристик:
  * энергетика аудио (громкость + всплески) — «яркость» момента;
  * текст Whisper — маркеры смеха/возбуждения, знаки препинания, капс;
  * опционально локальный LLM (Ollama) ранжирует «виральность».
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from .models import ClipCandidate, SceneBreak, TimestampedText

# --- Текстовые маркеры (мультиязычные RU/EN) ---

LAUGHTER_MARKERS = [
    "ха", "хах", "хех", "хых", "хихи", "аха", "гыг", "бугага",
    "смех", "ржёт", "лол", "lol", "lmao", "rofl", "haha", "hehe",
    "giggle", "[laughter]", "[смех]", "😂", "🤣",
]
HYPE_MARKERS = [
    "wow", "omg", "oh my", "no way", "what?!", "damn", "crazy",
    "боже", "нифига", "ничего себе", "офигеть", "офигел", "вау",
    "ого", "круто", "невероятно", "ужас", "шок", "сюрприз",
    "сумасшедший", "безумие", "охренеть", "жесть", "пипец",
    "отвал башки", "пушка", "зашквар", "топ", "имба",
]
STOP_MARKERS = [
    "музыка", "музыка играет", "реклама", "спонсор",
    "подпишись", "лайк", "не забудь",
]


def _word_in(text: str, words: Iterable[str]) -> int:
    low = text.lower()
    count = 0
    for w in words:
        if w in low:
            count += 1
    return count


def score_text_segment(seg: TimestampedText) -> float:
    """Оценка «интересности» текстового сегмента в диапазоне [0, 1]."""
    text = seg.text.strip()
    if not text:
        return 0.0
    low = text.lower()

    # Наказание за no_speech_prob (вероятно, шум/музыка)
    base = 0.1
    base += 0.9 * (1.0 - min(seg.no_speech_prob, 0.95))

    base += 0.5 * _word_in(text, LAUGHTER_MARKERS)
    base += 0.4 * _word_in(text, HYPE_MARKERS)

    # Знаки препинания и капс
    if any(p in text for p in "!?!…"):
        base += 0.15
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
    if uppercase_ratio > 0.4 and len(text) > 3:
        base += 0.2

    # Длина: очень короткие (<4 слов) или слишком длинные реплики менее ценны
    n_words = len(text.split())
    if n_words < 4:
        base -= 0.15
    if n_words > 45:
        base -= 0.1

    # СТОП-слова (реклама/музыка)
    if _word_in(text, STOP_MARKERS):
        base -= 0.5

    return float(np.clip(base, 0.0, 2.0))


def _snap_to_boundary(
    t: float,
    boundaries: List[float],
    direction: int,  # -1: влево (start), +1: вправо (end)
    max_shift: float = 2.5,
) -> float:
    """Подвигает границу к ближайшей смене сцены/тишине в пределах max_shift."""
    best = None
    for b in boundaries:
        delta = (b - t) * direction
        if 0 <= delta <= max_shift:
            if best is None or delta < best[0]:
                best = (delta, b)
    return best[1] if best else t


def group_segments(
    segments: List[TimestampedText],
    scene_breaks: List[SceneBreak],
    max_gap: float = 2.0,
) -> List[List[TimestampedText]]:
    """Склеивает сегменты транскрипции в «мысли» — законченные истории.

    Соседние реплики объединяются, пока пауза между ними мала и нет смены
    сцены. Большая пауза или смена сцены — граница мысли. Длина мысли
    ничем не ограничена — границы ставят только паузы и сцены.
    """
    scene_times = sorted(s.time for s in scene_breaks)
    groups: List[List[TimestampedText]] = []
    current: List[TimestampedText] = []

    def _break_here(prev_end: float, next_start: float) -> bool:
        gap = next_start - prev_end
        if gap > max_gap:
            return True
        for st in scene_times:
            if prev_end - 0.3 <= st <= next_start + 0.3:
                return True
        return False

    for seg in segments:
        if current:
            if _break_here(current[-1].end, seg.start):
                groups.append(current)
                current = [seg]
            else:
                current.append(seg)
        else:
            current = [seg]
    if current:
        groups.append(current)
    return groups


def _split_story(start: float, end: float, max_duration: float) -> List[tuple]:
    """Делит длинную историю на последовательные части без потери кусков."""
    if end - start <= max_duration:
        return [(start, end)]
    parts = []
    t = start
    while t < end - 1e-6:
        parts.append((t, min(end, t + max_duration)))
        t += max_duration
    return parts


def generate_candidates(
    energy_score: np.ndarray,
    energy_times: np.ndarray,
    text_segments: List[TimestampedText],
    scene_breaks: List[SceneBreak],
    silences: List[tuple],
    duration: float,
    min_duration: float = 4.0,
    max_duration: float = 180.0,
    max_gap: float = 2.0,
) -> List[ClipCandidate]:
    """Строит список кандидатов: базой служат «мысли» из сегментов Whisper.

    История целиком становится клипом; если она длиннее max_duration —
    разбивается на последовательные части. Границы притягиваются
    к тишине/сменам сцен для аккуратной нарезки.
    """
    boundaries = [s.time for s in scene_breaks]
    for start, end in silences:
        boundaries.extend([start, end])
    boundaries = sorted(set(round(b, 2) for b in boundaries))

    candidates: List[ClipCandidate] = []
    for group in group_segments(text_segments, scene_breaks, max_gap):
        start, end = group[0].start, group[-1].end
        # Расширяем до минимальной длины, притягивая границы к тишине/сценам
        if end - start < min_duration:
            need = min_duration - (end - start)
            start = max(0.0, start - need / 2)
            end = min(duration, end + need / 2)

        for part_start, part_end in _split_story(start, end, max_duration):
            part_start = _snap_to_boundary(part_start, boundaries, -1)
            part_end = _snap_to_boundary(part_end, boundaries, +1)
            if part_end - part_start < min_duration * 0.5:
                continue
            if part_end > duration:
                part_end = duration

            # Энергия окна
            mask = (energy_times >= part_start) & (energy_times <= part_end)
            e = float(np.mean(energy_score[mask])) if mask.any() else 0.0

            segs = [s for s in group if s.start >= part_start and s.end <= part_end]
            text = " ".join(s.text.strip() for s in segs).strip()
            cand = ClipCandidate(
                start=part_start,
                end=part_end,
                energy_score=e,
                text_score=float(np.max([score_text_segment(s) for s in segs])) if segs else 0.0,
                laughter_score=sum(_word_in(s.text, LAUGHTER_MARKERS) for s in segs),
                text=text,
                reason="story",
            )
            candidates.append(cand)

    return candidates


def _overlap(a: ClipCandidate, b: ClipCandidate) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def non_max_suppress(
    candidates: List[ClipCandidate],
    max_overlap: float = 0.5,
    min_gap: float = 1.5,
) -> List[ClipCandidate]:
    """Убирает пересекающиеся кандидаты, оставляя лучшие по score."""
    ranked = sorted(candidates, key=lambda c: c.total_score, reverse=True)
    picked: List[ClipCandidate] = []
    for cand in ranked:
        too_close = False
        for p in picked:
            ov = _overlap(cand, p) / max(1e-9, cand.duration)
            gap = min(abs(cand.start - p.end), abs(cand.end - p.start))
            if ov >= max_overlap or (ov > 0 and gap < min_gap):
                too_close = True
                break
        if not too_close:
            picked.append(cand)
    return picked


def rank_with_llm(
    candidates: List[ClipCandidate],
    model: str = "llama3.2",
    base_url: Optional[str] = None,
) -> List[ClipCandidate]:
    """Ранжирует кандидатов локальным LLM через Ollama (если запущен).

    Ошибки сети/модели не роняют пайплайн — кандидаты просто остаются
    со старым скором.
    """
    try:
        import ollama
    except ImportError:
        return candidates

    client = ollama.Client(host=base_url) if base_url else ollama
    for cand in candidates:
        if not cand.text:
            continue
        prompt = (
            "Ты — редактор виральных шортсов. Оцени момент в 0-10 по шкале "
            "«насколько он смешной/яркий/виральный» для TikTok/Shorts.\n"
            f"Реплика: {cand.text!r}\n"
            "Ответь только числом от 0 до 10."
        )
        try:
            resp = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_predict": 4},
            )
            raw = resp["message"]["content"].strip()
            num = float("".join(ch for ch in raw if ch.isdigit() or ch == ".")[:4])
            cand.llm_score = float(np.clip(num / 10.0, 0.0, 1.0))
        except Exception:
            cand.llm_score = None
    return candidates


def pick_top(
    candidates: List[ClipCandidate],
    max_clips: int = 10,
    min_score: float = 0.0,
) -> List[ClipCandidate]:
    """Финальный отбор: топ-N по суммарному скору."""
    ranked = sorted(candidates, key=lambda c: c.total_score, reverse=True)
    return [c for c in ranked if c.total_score >= min_score][:max_clips]