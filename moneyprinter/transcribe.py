"""Транскрипция видео через локальный Whisper (openai-whisper, бесплатно).

Импортируется лениво: если пакет не установлен, функция сообщает как поставить.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .models import TimestampedText

_MISSING_HINT = (
    "Whisper не установлен. Выполните: pip install 'moneyprinter[transcribe]' "
    "или pip install openai-whisper"
)


class TranscriptionError(RuntimeError):
    pass


def _load_model(model_name: str, device: str = "cpu"):
    try:
        import whisper
    except ImportError:
        raise TranscriptionError(_MISSING_HINT) from None
    return whisper.load_model(model_name, device=device)


def transcribe(
    audio_path: str,
    model_name: str = "base",
    device: str = "cpu",
    language: Optional[str] = None,
) -> List[TimestampedText]:
    """Транскрибирует аудио, возвращает сегменты с таймкодами."""
    model = _load_model(model_name, device)
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