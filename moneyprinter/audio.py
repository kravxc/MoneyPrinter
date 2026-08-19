"""Анализ аудио: громкость, тишина, "смеховые" энергетические всплески.

Работает полностью локально через numpy — без платных API.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .media import FFmpegError, _run


def read_audio(wav_path: str) -> Tuple[np.ndarray, int]:
    """Читает WAV (16k моно) в float32 массив [-1, 1]."""
    wav = np.fromfile(wav_path, dtype=np.int16)
    # Пропускаем 44-байтовый WAV-заголовок, если он есть
    if wav_path.lower().endswith(".wav") and len(wav) > 44:
        # заголовок чётный, поэтому сдвиг на 22 int16 == 44 байта
        wav = wav[22:]
    x = wav.astype(np.float32) / 32768.0
    return x, 16000


def loudness_envelope(
    x: np.ndarray, sr: int, win_sec: float = 0.25
) -> Tuple[np.ndarray, np.ndarray]:
    """Возвращает (rms, times): RMS-громкость по окнам и времена окон."""
    win = max(1, int(sr * win_sec))
    n = len(x)
    n_wins = max(1, n // win)
    x = x[: n_wins * win].reshape(n_wins, win)
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=1))
    times = (np.arange(n_wins) * win + win / 2) / sr
    return rms, times


def burstiness(x: np.ndarray, sr: int, win_sec: float = 0.25) -> np.ndarray:
    """Мера «дрожания» энергии — грубый прокси для смеха/возбуждённой речи.

    Высокая частотность перепадов амплитуды в окне.
    """
    win = max(1, int(sr * win_sec))
    n = len(x)
    n_wins = max(1, n // win)
    x = x[: n_wins * win].reshape(n_wins, win)
    diff = np.abs(np.diff(x, axis=1))
    return np.mean(diff, axis=1)


def silence_gaps(
    x: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    min_silence_sec: float = 0.5,
    win_sec: float = 0.05,
) -> List[Tuple[float, float]]:
    """Находит интервалы тишины — удобные границы для нарезки.

    Возвращает список (start, end) в секундах.
    """
    rms, times = loudness_envelope(x, sr, win_sec)
    threshold = 10 ** (threshold_db / 20.0)
    silent = rms < threshold
    gaps: List[Tuple[float, float]] = []
    in_gap = False
    gap_start = 0.0
    for is_silent, t in zip(silent, times):
        if is_silent and not in_gap:
            in_gap = True
            gap_start = t - win_sec / 2
        elif not is_silent and in_gap:
            in_gap = False
            gap_end = t - win_sec / 2
            if gap_end - gap_start >= min_silence_sec:
                gaps.append((gap_start, gap_end))
    if in_gap:
        gap_end = times[-1] + win_sec / 2
        if gap_end - gap_start >= min_silence_sec:
            gaps.append((gap_start, gap_end))
    return gaps


def energy_score_window(
    x: np.ndarray,
    sr: int,
    win_sec: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Нормализованный скоринг «яркости» по окнам: RMS + всплески.

    Возвращает (score, times). score в [0, 1].
    """
    rms, times = loudness_envelope(x, sr, win_sec)
    burst = burstiness(x, sr, win_sec)
    rms_norm = _normalize(rms)
    burst_norm = _normalize(burst)
    score = 0.7 * rms_norm + 0.3 * burst_norm
    return score, times


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, 10), np.percentile(arr, 95)
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def analyze_audio(
    wav_path: str,
    win_sec: float = 1.0,
) -> dict:
    """Сводный анализ аудио для дальнейшего скоринга."""
    x, sr = read_audio(wav_path)
    score, times = energy_score_window(x, sr, win_sec)
    silences = silence_gaps(x, sr)
    return {
        "samples": x,
        "sample_rate": sr,
        "duration": len(x) / sr,
        "energy_score": score,
        "energy_times": times,
        "silences": silences,
    }