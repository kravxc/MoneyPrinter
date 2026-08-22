"""Режим «сериал/фильм → микро-серия в TikTok».

В отличие от обычного process (где ищем яркие моменты и ранжируем по скору),
здесь видео режется **подряд, по порядку** на равные части заданной длины.
Одна серия сериала так превращается в серию микро-роликов для TikTok, которые
выглядят как продолжение (часть 1/12, 2/12, …). Отбора по «виральности» нет —
режем всё подряд.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import cutting
from . import hashtags as hashtags_mod
from . import media
from .models import ClipCandidate, ClipResult, PipelineResult


@dataclass
class SerialConfig:
    input_path: str
    output_dir: str = "clips"
    part_duration: float = 60.0       # длина одной микро-серии, сек
    max_parts: int = 0                # 0 = все части до конца видео
    start: float = 0.0                # с какого момента резать (сек)
    end: float = 0.0                  # до какого момента (0 = до конца)
    vertical: bool = True
    blur_bg: bool = True
    series_title: str = ""            # название сериала (для подписи/тегов)
    episode: int = 1                  # номер серии
    base_hashtags: list = None        # доп. теги (название сериала и т.п.)
    jobs: int = 0                     # 0 = все ядра


def _build_parts(cfg: SerialConfig, duration: float) -> list:
    """Режет [start, end] на последовательные части по part_duration.

    Возвращает список (part_index, start, end), индексация с 1, строго по порядку.
    """
    start = max(0.0, cfg.start)
    end = cfg.end if cfg.end and cfg.end > start else duration
    end = min(end, duration)
    if end - start < cfg.part_duration:
        # видео короче одной части — один клип на всё
        return [(1, start, end)]

    parts = []
    i = 1
    t = start
    while t < end - 1e-6:
        p_end = min(end, t + cfg.part_duration)
        # последний кусок короче part_duration — оставляем как есть
        parts.append((i, t, p_end))
        t += cfg.part_duration
        i += 1
    if cfg.max_parts and cfg.max_parts > 0:
        parts = parts[: cfg.max_parts]
    return parts


def _make_caption_and_tags(cfg: SerialConfig, part_idx: int, total: int, duration: float) -> tuple:
    """Собирает подпись и теги для микро-серии.

    В подпись добавляется «Серия N | Часть X/Y», чтобы TikTok-лента
    воспринималась как продолжение. Теги = базовый набор + название сериала.
    """
    title_bit = cfg.series_title.strip() if cfg.series_title else "Сериал"
    header = f"{title_bit} | Серия {cfg.episode} | Часть {part_idx}/{total}"
    snippet = f"⏱ {duration:.0f} сек. Продолжение — следующим роликом 👉"
    tags_text = f"{title_bit} серия {cfg.episode} часть {part_idx}"
    tags = hashtags_mod.generate_hashtags(
        tags_text, base=cfg.base_hashtags or []
    )
    caption = f"{header}\n\n{snippet}\n\n" + " ".join(f"#{t}" for t in tags)
    return caption, tags


def process_serial(cfg: SerialConfig) -> PipelineResult:
    media.require_ffmpeg()
    input_path = str(cfg.input_path)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = media.probe(input_path)
    result = PipelineResult(input_path=input_path, duration=info.duration)
    jobs = cfg.jobs or os.cpu_count() or 1

    parts = _build_parts(cfg, info.duration)
    total = len(parts)
    if total == 0:
        return result

    # Считаем хронометраж для единого прогресс-бара
    total_dur = sum(e - s for _, s, e in parts)
    bar = tqdm(total=total_dur, desc="Нарезка серии", unit="s")
    prefixes = []
    acc = 0.0
    for _, s, e in parts:
        prefixes.append(acc)
        acc += e - s

    def _worker(idx: int, start: float, end: float, prefix: float):
        cand = ClipCandidate(start=start, end=end, text=f"{cfg.series_title} часть {idx}")
        caption, tags = _make_caption_and_tags(cfg, idx, total, end - start)
        out_name = f"clip_{idx:02d}_s{start:.1f}_e{end:.1f}.mp4"
        out_path = str(out_dir / out_name)
        if cfg.vertical:
            cutting.make_vertical(input_path, cand, out_path, blur_bg=cfg.blur_bg, bar=bar, offset=prefix)
        else:
            cutting.cut_clip(input_path, cand, out_path, bar=bar, offset=prefix)
        return ClipResult(
            path=out_path, start=start, end=end, duration=end - start,
            score=0.0, text=cand.text, reason="serial", vertical=cfg.vertical,
            hashtags=tags, caption=caption,
        )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    clips = [None] * len(parts)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_worker, idx, s, e, prefixes[i]): i
            for i, (idx, s, e) in enumerate(parts)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            clip = fut.result()
            clips[i] = clip
            print(f"  ✓ {Path(clip.path).name}  [{clip.start:7.1f}s → {clip.end:7.1f}s]  «{clip.caption[:55]}»")
    bar.close()

    result.clips = [c for c in clips if c is not None]
    # уже по порядку (индексы по частям), но на всякий сортируем по start
    result.clips.sort(key=lambda c: c.start)

    # Отчёт
    report = out_dir / "report.json"

    def _to_jsonable(obj):
        if hasattr(obj, "__dict__"):
            return {k: _to_jsonable(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        return obj

    report.write_text(json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    import csv
    csv_path = out_dir / "report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "start", "end", "duration", "text", "hashtags", "caption"])
        for c in result.clips:
            writer.writerow([c.path, f"{c.start:.2f}", f"{c.end:.2f}", f"{c.duration:.2f}", c.text,
                             " ".join(f"#{t}" for t in c.hashtags), c.caption])

    return result
