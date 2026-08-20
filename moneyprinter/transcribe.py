"""Транскрипция видео.

Бэкенд — faster-whisper (CTranslate2): те же веса Whisper, но в ~4 раза
быстрее openai-whisper на CPU при том же качестве. При наличии NVIDIA GPU
автоматически используется CUDA. На CPU длинные ролики транскрибируются
параллельно по чанкам (каждый кусок — отдельный процесс) с дедупликацией
стыков — ещё ~3-4x к скорости без потери качества.

Если faster-whisper не установлен — он подтягивается сам (pip install).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import wave
from typing import List, Optional, Tuple

from tqdm import tqdm

from .models import TimestampedText

_MISSING_HINT = (
    "Whisper не установлен. Выполните: pip install 'moneyprinter[transcribe]' "
    "или pip install faster-whisper"
)

# Параметры чанкования (сек)
CHUNK_SEC = 300.0
CHUNK_OVERLAP = 15.0
CHUNK_MIN_DURATION = 300.0  # меньше — одиночный проход


class TranscriptionError(RuntimeError):
    pass


def _ensure_faster_whisper(auto_install: bool) -> bool:
    """Проверяет наличие faster-whisper; при auto_install сам ставит его."""
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        pass
    if not auto_install:
        return False
    print("[i] faster-whisper не найден — устанавливаю (первый раз может занять пару минут)...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "faster-whisper"]
        )
        import faster_whisper  # noqa: F401

        return True
    except Exception:
        print("[warn] Не удалось установить faster-whisper — пробую openai-whisper")
        return False


# --- CUDA-поддержка ---------------------------------------------------------

def _has_nvidia_gpu() -> bool:
    """Определяет наличие NVIDIA GPU без тяжёлых зависимостей."""
    import shutil

    try:
        if shutil.which("nvidia-smi"):
            result = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
            )
            return result.returncode == 0
    except OSError:
        pass
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _resolve_device(device: str) -> str:
    """'auto' → cuda при наличии NVIDIA GPU, иначе cpu."""
    if device in ("cuda", "cpu"):
        return device
    if _has_nvidia_gpu():
        print("[i] Обнаружена NVIDIA GPU — транскрипция на CUDA")
        return "cuda"
    return "cpu"


def _cuda_libs_missing() -> bool:
    """Проверяет, доступны ли cuBLAS/cuDNN для CTranslate2."""
    import ctypes

    try:
        if sys.platform == "win32":
            ctypes.WinDLL("cublas64_12.dll")
            ctypes.WinDLL("cudnn64_9.dll")
        else:
            ctypes.CDLL("libcublas.so.12")
            ctypes.CDLL("libcudnn.so.9")
        return False
    except OSError:
        return True


def _ensure_cuda_libs() -> bool:
    """Доустанавливает cuBLAS/cuDNN (pip) и добавляет их bin в PATH."""
    if not _cuda_libs_missing():
        return True

    print("[i] CUDA-библиотеки (cuBLAS/cuDNN) не найдены — устанавливаю через pip...")
    try:
        subprocess.check_call(
            [
                sys.executable, "-m", "pip", "install", "--quiet",
                "nvidia-cublas-cu12", "nvidia-cudnn-cu12",
            ]
        )
    except Exception:
        return False

    import glob
    import site

    for p in site.getsitepackages():
        if sys.platform == "win32":
            for d in glob.glob(os.path.join(p, "nvidia", "*", "bin")):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        else:
            for d in glob.glob(os.path.join(p, "nvidia", "*", "lib")):
                os.environ["LD_LIBRARY_PATH"] = (
                    d + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
                )
    return not _cuda_libs_missing()


# --- Чанки аудио ------------------------------------------------------------

def _make_chunks(
    wav_path: str,
    tmpdir: str,
    chunk_sec: float = CHUNK_SEC,
    overlap_sec: float = CHUNK_OVERLAP,
) -> List[Tuple[str, float, float]]:
    """Режет WAV на чанки с перекрытием.

    Возвращает список (path, offset, coverage): offset — сдвиг от начала
    исходника, coverage — длительность чанка для прогресс-бара.
    """
    with wave.open(wav_path, "rb") as w:
        framerate = w.getframerate()
        channels = w.getnchannels()
        total_frames = w.getnframes()
    if framerate <= 0:
        return [(wav_path, 0.0, 0.0)]

    total_dur = total_frames / framerate
    step_frames = max(1, int((chunk_sec - overlap_sec) * framerate))
    chunk_frames = int(chunk_sec * framerate)

    chunks: List[Tuple[str, float, float]] = []
    start_frame = 0
    i = 0
    while start_frame < total_frames:
        out = os.path.join(tmpdir, f"chunk_{i:04d}.wav")
        n_frames = min(chunk_frames, total_frames - start_frame)
        with wave.open(wav_path, "rb") as src, wave.open(out, "wb") as dst:
            dst.setnchannels(channels)
            dst.setsampwidth(2)
            dst.setframerate(framerate)
            src.setpos(start_frame)
            dst.writeframes(src.readframes(n_frames))
        offset = start_frame / framerate
        chunks.append((out, offset, n_frames / framerate))
        start_frame += step_frames
        i += 1
    return chunks


# --- Параллельные воркеры (модульный уровень для spawn/picklable) -----------

_worker_model = None
_worker_language = None


def _init_worker(model_name: str, language: Optional[str]) -> None:
    global _worker_model, _worker_language
    from faster_whisper import WhisperModel

    _worker_model = WhisperModel(model_name, device="cpu", compute_type="int8")
    _worker_language = language


def _transcribe_chunk(task: Tuple[str, float, float]) -> List[Tuple[float, float, str, float]]:
    """Транскрибирует один чанк; возвращает сегменты в абсолютном времени."""
    wav_path, offset, _coverage = task
    segs, _ = _worker_model.transcribe(
        wav_path,
        vad_filter=True,
        language=_worker_language,
    )
    return [
        (offset + float(s.start), offset + float(s.end), s.text, float(s.no_speech_prob))
        for s in segs
        if (s.text or "").strip()
    ]


def _merge_chunk_results(
    results: List[List[Tuple[float, float, str, float]]],
    chunks: List[Tuple[str, float, float]],
    overlap_sec: float = CHUNK_OVERLAP,
) -> List[TimestampedText]:
    """Склеивает результаты чанков, убирая дубликаты из зоны перекрытия."""
    merged: List[TimestampedText] = []
    for segs, (_, offset, _) in zip(results, chunks):
        cut = offset + overlap_sec if offset > 0 else 0.0
        for start, end, text, nsp in segs:
            if start < cut:
                continue
            merged.append(
                TimestampedText(start=start, end=end, text=text, no_speech_prob=nsp)
            )
    return merged


def _transcribe_chunked_parallel(
    audio_path: str,
    model_name: str,
    language: Optional[str],
    duration: float,
    jobs: int,
) -> List[TimestampedText]:
    """Транскрипция длинного аудио на CPU: чанки по ядрам."""
    print(f"[i] Транскрипция параллельно ({jobs} потоков)...")
    with tempfile.TemporaryDirectory(prefix="moneyprinter_chunks_") as td:
        chunks = _make_chunks(audio_path, td)
        ctx = mp.get_context("spawn")
        bar = tqdm(total=duration, desc="Транскрипция", unit="s")
        try:
            with ctx.Pool(
                processes=jobs,
                initializer=_init_worker,
                initargs=(model_name, language),
            ) as pool:
                results = []
                for result in pool.imap(_transcribe_chunk, chunks):
                    results.append(result)
                    bar.n = min(bar.n + CHUNK_SEC, duration)
                    bar.refresh()
        finally:
            bar.close()
        return _merge_chunk_results(results, chunks)


# --- Одиночный проход -------------------------------------------------------

def _transcribe_faster_whisper(
    audio_path: str,
    model_name: str,
    device: str,
    language: Optional[str],
    duration: Optional[float] = None,
    jobs: int = 1,
) -> List[TimestampedText]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise TranscriptionError(_MISSING_HINT) from None

    device = _resolve_device(device)
    if device == "cuda" and not _ensure_cuda_libs():
        print("[warn] CUDA-библиотеки недоступны — транскрипция на CPU")
        device = "cpu"

    if device == "cpu" and jobs > 1 and duration and duration > CHUNK_MIN_DURATION:
        try:
            return _transcribe_chunked_parallel(
                audio_path, model_name, language, duration, jobs
            )
        except Exception as exc:
            print(f"[warn] Параллельная транскрипция не удалась ({exc}) — одиночный проход")

    compute_type = "float16" if device == "cuda" else "int8"
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        kwargs = {"vad_filter": True}
        if language:
            kwargs["language"] = language
        segments_iter, _ = model.transcribe(audio_path, **kwargs)

        bar = tqdm(
            total=duration if duration and duration > 0 else None,
            desc="Транскрипция",
            unit="s",
        )
        segments: List[TimestampedText] = []
        try:
            for seg in segments_iter:
                if duration and duration > 0:
                    bar.n = min(float(seg.end), duration)
                    bar.refresh()
                else:
                    bar.update(1)
                text = (seg.text or "").strip()
                if not text:
                    continue
                segments.append(
                    TimestampedText(
                        start=float(seg.start),
                        end=float(seg.end),
                        text=text,
                        no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0)),
                    )
                )
        finally:
            bar.close()
        return segments
    except Exception as exc:
        if device == "cuda":
            print(f"[warn] Ошибка CUDA ({exc}) — повторяю на CPU")
            return _transcribe_faster_whisper(
                audio_path, model_name, "cpu", language, duration, jobs
            )
        raise


# --- openai-whisper фолбэк --------------------------------------------------

def _transcribe_openai_whisper(
    audio_path: str,
    model_name: str,
    device: str,
    language: Optional[str],
) -> List[TimestampedText]:
    try:
        import whisper
    except ImportError:
        raise TranscriptionError(_MISSING_HINT) from None

    device = _resolve_device(device)
    model = whisper.load_model(model_name, device=device)
    kwargs = {"fp16": False}
    if language:
        kwargs["language"] = language
    result = model.transcribe(audio_path, **kwargs)

    segments: List[TimestampedText] = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            TimestampedText(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=text,
                no_speech_prob=float(seg.get("no_speech_prob", 0.0)),
            )
        )
    return segments


def transcribe(
    audio_path: str,
    model_name: str = "base",
    device: str = "auto",
    language: Optional[str] = None,
    auto_install: bool = True,
    duration: Optional[float] = None,
    jobs: int = 1,
) -> List[TimestampedText]:
    """Транскрибирует аудио, возвращает сегменты с таймкодами."""
    if _ensure_faster_whisper(auto_install):
        return _transcribe_faster_whisper(
            audio_path, model_name, device, language, duration, jobs
        )
    try:
        return _transcribe_openai_whisper(audio_path, model_name, device, language)
    except Exception as exc:
        raise TranscriptionError(_MISSING_HINT) from exc