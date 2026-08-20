"""Транскрипция видео.

Бэкенд по умолчанию — faster-whisper (CTranslate2): те же веса Whisper,
но в ~4 раза быстрее на CPU при том же качестве. Если есть NVIDIA GPU —
автоматически используется CUDA. Если faster-whisper не установлен —
он подтягивается сам (pip install), фолбэк на openai-whisper.
"""

from __future__ import annotations

import subprocess
import sys
from typing import List, Optional

from .models import TimestampedText

_MISSING_HINT = (
    "Whisper не установлен. Выполните: pip install 'moneyprinter[transcribe]' "
    "или pip install faster-whisper"
)


class TranscriptionError(RuntimeError):
    pass


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


def _transcribe_faster_whisper(
    audio_path: str,
    model_name: str,
    device: str,
    language: Optional[str],
) -> List[TimestampedText]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise TranscriptionError(_MISSING_HINT) from None

    device = _resolve_device(device)
    compute_type = "float16" if device == "cuda" else "int8"
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as exc:
        if device == "cuda":
            print(f"[warn] CUDA не запустился ({exc}) — возвращаюсь на CPU")
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
        else:
            raise

    kwargs = {"vad_filter": True}
    if language:
        kwargs["language"] = language
    segments_iter, _ = model.transcribe(audio_path, **kwargs)

    segments: List[TimestampedText] = []
    for seg in segments_iter:
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
    return segments


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
) -> List[TimestampedText]:
    """Транскрибирует аудио, возвращает сегменты с таймкодами."""
    if _ensure_faster_whisper(auto_install):
        return _transcribe_faster_whisper(audio_path, model_name, device, language)
    try:
        return _transcribe_openai_whisper(audio_path, model_name, device, language)
    except Exception as exc:
        raise TranscriptionError(_MISSING_HINT) from exc