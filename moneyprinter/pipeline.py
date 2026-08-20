"""Оркестрация всего пайплайна: анализ → кандидаты → нарезка → отчёт."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from . import audio as audio_mod
from . import cutting
from . import media
from . import score as score_mod
from . import scenes as scenes_mod
from . import transcribe as transcribe_mod
from .models import ClipCandidate, ClipResult, PipelineResult, TimestampedText


@dataclass
class Config:
    input_path: str
    output_dir: str = "clips"
    max_clips: int = 10
    min_duration: float = 10.0
    max_duration: float = 180.0
    min_score: float = 0.0
    vertical: bool = True
    blur_bg: bool = True
    story_gap: float = 2.0
    remove_ads: bool = True  # вырезать банеры казино/беттинга из клипов
    ocr_interval: Optional[float] = None  # шаг сэмплирования кадров для OCR, сек
    whisper_model: str = "base"
    device: str = "auto"
    language: Optional[str] = None
    scene_threshold: float = 27.0
    llm_model: Optional[str] = None
    llm_url: Optional[str] = None
    jobs: int = 0  # 0 = все ядра CPU
    auto_install: bool = True  # сам доустанавливать недостающие AI-зависимости
    keep_audio: bool = False  # не используется сейчас, задел на будущее
    temp_dir: Optional[str] = None


def _energy_only_candidates(
    energy_score: np.ndarray,
    energy_times: np.ndarray,
    duration: float,
    min_duration: float,
    max_duration: float,
    max_clips: int,
) -> List[ClipCandidate]:
    """Фолбэк без Whisper: кандидаты вокруг локальных пиков энергии."""
    n = len(energy_score)
    if n == 0:
        return []
    cands: List[ClipCandidate] = []
    # окно = типичная длина клипа
    win = max(1, int(min_duration * (n / duration))) if duration else 1
    for i in range(0, n, max(1, win // 2)):
        seg = energy_score[i : i + win]
        if len(seg) == 0:
            continue
        peak = int(np.argmax(seg)) + i
        start = energy_times[peak] - min_duration / 2
        end = start + min_duration
        if start < 0 or end > duration:
            continue
        e = float(np.mean(seg))
        cands.append(ClipCandidate(start=start, end=end, energy_score=e, reason="energy-only"))
    return cands


def process(cfg: Config) -> PipelineResult:
    media.require_ffmpeg()
    input_path = str(cfg.input_path)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = media.probe(input_path)
    result = PipelineResult(input_path=input_path, duration=info.duration)
    jobs = cfg.jobs or os.cpu_count() or 1

    with tempfile.TemporaryDirectory(prefix="moneyprinter_") as tmp:
        tmp = cfg.temp_dir or tmp
        wav_path = None
        audio = None

        # 1) Аудио-анализ (если есть аудиодорожка)
        if info.has_audio:
            wav_path = media.decode_audio_wav(input_path, str(Path(tmp) / "audio.wav"))
            audio = audio_mod.analyze_audio(wav_path)
        else:
            # без аудио — только сцены + равномерные кандидаты
            audio = {
                "samples": np.array([], dtype=np.float32),
                "sample_rate": 16000,
                "duration": info.duration,
                "energy_score": np.zeros(1),
                "energy_times": np.array([info.duration / 2]),
                "silences": [],
            }

        # 2) Сцены
        scene_breaks = scenes_mod.detect_scenes(
            input_path, duration=info.duration, threshold=cfg.scene_threshold, prefer_pyscenedetect=True
        )
        scene_times = [s.time for s in scene_breaks]

        # 3) Транскрипция
        text_segments: List[TimestampedText] = []
        if wav_path:
            try:
                text_segments = transcribe_mod.transcribe(
                    wav_path,
                    model_name=cfg.whisper_model,
                    device=cfg.device,
                    language=cfg.language,
                    auto_install=cfg.auto_install,
                    duration=info.duration,
                    jobs=jobs,
                )
            except transcribe_mod.TranscriptionError as exc:
                print(f"[warn] Транскрипция недоступна ({exc}). Использую эвристики без текста.")

        # 3.5) Детекция визуальных рекламных баннеров (казино/беттинг) через OCR
        banner_ranges: List = []
        if cfg.remove_ads and text_segments:
            from . import banner as banner_mod

            if banner_mod.ensure_ocr(cfg.auto_install):
                banner_ranges = banner_mod.detect_banner_ranges(
                    input_path, info.duration, cfg.ocr_interval, jobs
                )
                if banner_ranges:
                    banner_mod.mark_segments_by_ranges(text_segments, banner_ranges)
                    n_ads = sum(1 for s in text_segments if s.is_ad)
                    print(
                        f"[i] Рекламные баннеры на {len(banner_ranges)} участках — "
                        f"затронуто {n_ads} фрагментов, они будут вырезаны"
                    )
                else:
                    print("[i] Рекламных баннеров не обнаружено")
            else:
                print("[warn] OCR недоступен — визуальные баннеры не детектируются")

        # 4) Кандидаты
        if text_segments:
            candidates = score_mod.generate_candidates(
                audio["energy_score"],
                audio["energy_times"],
                text_segments,
                scene_breaks,
                audio["silences"],
                duration=info.duration,
                min_duration=cfg.min_duration,
                max_duration=cfg.max_duration,
                max_gap=cfg.story_gap,
                remove_ads=cfg.remove_ads,
                banner_ranges=banner_ranges,
            )
        else:
            candidates = _energy_only_candidates(
                audio["energy_score"],
                audio["energy_times"],
                info.duration,
                cfg.min_duration,
                cfg.max_duration,
                cfg.max_clips,
            )

        # 5) Ранжирование
        if cfg.llm_model:
            candidates = score_mod.rank_with_llm(
                candidates, model=cfg.llm_model, base_url=cfg.llm_url
            )
        candidates = score_mod.non_max_suppress(candidates)
        picked = score_mod.pick_top(candidates, max_clips=cfg.max_clips, min_score=cfg.min_score)

        # 6) Нарезка (параллельно, чтобы задействовать все ядра CPU)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _cut_worker(cand: ClipCandidate, index: int, prefix: float):
            out_name = f"clip_{index:02d}_s{cand.start:.1f}_e{cand.end:.1f}.mp4"
            out_path = str(out_dir / out_name)
            if cfg.vertical:
                cutting.make_vertical(
                    input_path, cand, out_path, blur_bg=cfg.blur_bg, bar=bar, offset=prefix
                )
            else:
                cutting.cut_clip(input_path, cand, out_path, bar=bar, offset=prefix)
            return cand, out_name, out_path

        bar = tqdm(total=sum(c.duration for c in picked), desc="Нарезка", unit="s")
        prefixes = []
        acc = 0.0
        for c in picked:
            prefixes.append(acc)
            acc += c.duration

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [
                pool.submit(_cut_worker, cand, i, prefixes[i - 1])
                for i, cand in enumerate(picked, start=1)
            ]
            for fut in as_completed(futures):
                cand, out_name, out_path = fut.result()
                result.clips.append(
                    ClipResult(
                        path=out_path,
                        start=cand.start,
                        end=cand.end,
                        duration=cand.duration,
                        score=cand.total_score,
                        text=cand.text,
                        reason=cand.reason,
                        vertical=cfg.vertical,
                    )
                )
                print(
                    f"  ✓ {out_name}  [{cand.start:7.1f}s → {cand.end:7.1f}s]  "
                    f"score={cand.total_score:.2f}  «{cand.text[:60]}»"
                )
        bar.close()

        # 7) Отчёт
        report = out_dir / "report.json"

        def _to_jsonable(obj):
            if hasattr(obj, "__dict__"):
                return {k: _to_jsonable(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(v) for v in obj]
            return obj

        report.write_text(
            json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        csv_path = out_dir / "report.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "start", "end", "duration", "score", "text"])
            for c in result.clips:
                writer.writerow([c.path, f"{c.start:.2f}", f"{c.end:.2f}", f"{c.duration:.2f}", f"{c.score:.3f}", c.text])

    return result