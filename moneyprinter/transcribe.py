"""Транскрипция видео.

Бэкенд по умолчанию — faster-whisper (CTranslate2): те же веса Whisper,
но в ~4 раза быстрее на CPU при том же качестве. Если faster-whisper
не установлен — фолбэк на openai-whisper.
"""

from __future__ import annotations

from typing import List, Optional

from .models import TimestampedText

_MISSING_HINT = (
    "Whisper не установлен. Выполните: pip install 'moneyprinter[transcribe]' "
    "или pip install faster-whisper"
)


class TranscriptionError(RuntimeError):
    pass


def _resolve_device(device: str) -> str:
    """'auto' → cpu (faster-whisper на CPU самый надёжный и быстрый)."""
    if device in ("auto", "mps"):
        return "cpu"
    return device


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
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
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
) -> List[TimestampedText]:
    """Транскрибирует аудио, возвращает сегменты с таймкодами."""
    try:
        return _transcribe_faster_whisper(audio_path, model_name, device, language)
    except TranscriptionError:
        # faster-whisper не установлен — пробуем openai-whisper
        try:
            return _transcribe_openai_whisper(audio_path, model_name, device, language)
        except TranscriptionError:
            raise
        except Exception as exc:  # openai-whisper не установлен
            raise TranscriptionError(_MISSING_HINT) from exc