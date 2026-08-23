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
from typing import Optional

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
    whisper_model: str = "base"       # модель транскрипции
    device: str = "auto"
    language: Optional[str] = None
    llm_model: Optional[str] = None   # локальная LLM для описания по содержанию
    llm_url: Optional[str] = None
    auto_install: bool = True
    transcribe_audio: bool = True     # если False — описания без крючка по содержанию


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


def _make_caption_and_tags(cfg: SerialConfig, part_idx: int, total: int, duration: float, text: str = "") -> tuple:
    """Собирает подпись и теги для микро-серии.

    Описание — по содержанию (крючок из транскрипции через LLM/эвристику),
    а не одинаковое на все части. Теги = базовый набор + название сериала
    + тематика по тексту.
    """
    title_bit = cfg.series_title.strip() if cfg.series_title else "Сериал"
    header = f"{title_bit} | Серия {cfg.episode} | Часть {part_idx}"
    hook = hashtags_mod.generate_hook(
        text, llm_model=cfg.llm_model, llm_url=cfg.llm_url, limit=70
    )
    snippet = (hook + "\n\n") if hook else ""
    snippet += f"⏱ {duration:.0f} сек. Продолжение — следующим роликом 👉"
    # теги: по названию сериала + по содержанию части
    tags_text = f"{title_bit} серия {cfg.episode} {text}"
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

    # --- Транскрипция всего видео один раз (для описания по содержанию) ---
    # Сопоставим каждой части её текст по таймкодам.
    part_texts: dict = {}
    if cfg.transcribe_audio and info.has_audio:
        try:
            from . import transcribe as transcribe_mod
            import tempfile

            with tempfile.TemporaryDirectory(prefix="moneyprinter_serial_") as tmp:
                wav = media.decode_audio_wav(input_path, str(Path(tmp) / "audio.wav"))
                segs = transcribe_mod.transcribe(
                    wav,
                    model_name=cfg.whisper_model,
                    device=cfg.device,
                    language=cfg.language,
                    auto_install=cfg.auto_install,
                    duration=info.duration,
                    jobs=jobs,
                )
            for idx, s, e in parts:
                # собираем текст сегментов, попадающих в [s, e]
                piece = " ".join(
                    seg.text.strip() for seg in segs if seg.start >= s and seg.end <= e
                )
                part_texts[idx] = piece.strip()
            print(f"[i] Транскрибировано сегментов: {len(segs)}; текст разложен по {total} частям.")
        except Exception as exc:
            print(f"[warn] Транскрипция недоступна ({exc}). Описания будут без крючка по содержанию.")
    else:
        if not cfg.transcribe_audio:
            print("[i] Транскрипция отключена (--no-audio-desc) — описания без крючка по содержанию.")
        else:
            print("[warn] Нет аудиодорожки — описания по содержанию не будут сгенерированы.")

    # Считаем хронометраж для единого прогресс-бара
    total_dur = sum(e - s for _, s, e in parts)
    bar = tqdm(total=total_dur, desc="Нарезка серии", unit="s")
    prefixes = []
    acc = 0.0
    for _, s, e in parts:
        prefixes.append(acc)
        acc += e - s

    def _worker(idx: int, start: float, end: float, prefix: float):
        text = part_texts.get(idx, "")
        cand = ClipCandidate(start=start, end=end, text=text)
        caption, tags = _make_caption_and_tags(cfg, idx, total, end - start, text=text)
        out_name = f"clip_{idx:02d}_s{start:.1f}_e{end:.1f}.mp4"
        out_path = str(out_dir / out_name)
        if cfg.vertical:
            cutting.make_vertical(input_path, cand, out_path, blur_bg=cfg.blur_bg, bar=bar, offset=prefix)
        else:
            cutting.cut_clip(input_path, cand, out_path, bar=bar, offset=prefix)
        return ClipResult(
            path=out_path, start=start, end=end, duration=end - start,
            score=0.0, text=text, reason="serial", vertical=cfg.vertical,
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


def regenerate_captions(
    input_path: str,
    output_dir: str = "clips",
    series_title: str = "",
    episode: int = 1,
    base_hashtags: list = None,
    whisper_model: str = "base",
    device: str = "auto",
    language: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_url: Optional[str] = None,
    auto_install: bool = True,
    jobs: int = 0,
) -> int:
    """Перегенерирует описания/хештеги для уже нарезанных клипов.

    Читает report.json в output_dir (там лежат start/end каждого клипа),
    заново транскрибирует исходник и переписывает caption+hashtags у каждого
    клипа, сохраняя сами видеофайлы. Возвращает число обновлённых клипов.
    """
    import csv
    import json as _json
    import tempfile

    from . import transcribe as transcribe_mod

    out_dir = Path(output_dir)
    report = out_dir / "report.json"
    if not report.exists():
        raise RuntimeError(f"Не найден {report}. Сначала выполните `moneyprinter serial`.")

    data = _json.loads(report.read_text(encoding="utf-8"))
    clips = data.get("clips", [])
    if not clips:
        print("[i] В отчёте нет клипов.")
        return 0

    info = media.probe(input_path)
    jobs = jobs or os.cpu_count() or 1

    # Переписываем конфиг из данных клипов (если есть series_title и т.п.)
    cfg = SerialConfig(
        input_path=input_path,
        output_dir=output_dir,
        series_title=series_title or "",
        episode=episode,
        base_hashtags=base_hashtags,
        whisper_model=whisper_model,
        device=device,
        language=language,
        llm_model=llm_model,
        llm_url=llm_url,
        auto_install=auto_install,
    )

    # Транскрипция один раз
    print("[i] Транскрибирую исходник для перегенерации описаний...")
    with tempfile.TemporaryDirectory(prefix="moneyprinter_regen_") as tmp:
        wav = media.decode_audio_wav(input_path, str(Path(tmp) / "audio.wav"))
        segs = transcribe_mod.transcribe(
            wav, model_name=whisper_model, device=device, language=language,
            auto_install=auto_install, duration=info.duration, jobs=jobs,
        )

    def _text_for(start: float, end: float) -> str:
        return " ".join(
            s.text.strip() for s in segs if s.start >= start and s.end <= end
        ).strip()

    total = len(clips)
    for cli in clips:
        start, end = float(cli["start"]), float(cli["end"])
        text = _text_for(start, end)
        # для подписи нужен реальный номер части — берём из имени файла clip_XX
        import re as _re
        m = _re.search(r"clip_(\d+)", Path(cli["path"]).name)
        part_idx = int(m.group(1)) if m else 0
        cap, tags = _make_caption_and_tags(cfg, part_idx, total, end - start, text=text)
        cli["text"] = text
        cli["hashtags"] = tags
        cli["caption"] = cap

    # Сохраняем обратно
    report.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out_dir / "report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "start", "end", "duration", "text", "hashtags", "caption"])
        for c in clips:
            writer.writerow([c["path"], f"{c['start']:.2f}", f"{c['end']:.2f}", f"{c['duration']:.2f}",
                             c.get("text", ""), " ".join(f"#{t}" for t in c.get("hashtags", [])), c.get("caption", "")])

    print(f"[✓] Обновлено описаний: {total} → {output_dir}")
    return total
