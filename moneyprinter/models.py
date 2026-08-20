"""Структуры данных проекта."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VideoInfo:
    """Метаданные входного видео."""

    path: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


@dataclass
class TimestampedText:
    """Сегмент транскрипции Whisper."""

    start: float
    end: float
    text: str
    no_speech_prob: float = 0.0
    is_ad: bool = False


@dataclass
class SceneBreak:
    """Точка смены сцены (кадр)."""

    time: float


@dataclass
class ClipCandidate:
    """Кандидат на клип: момент + скоринги."""

    start: float
    end: float
    energy_score: float = 0.0
    text_score: float = 0.0
    laughter_score: float = 0.0
    llm_score: Optional[float] = None
    reason: str = ""
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def total_score(self) -> float:
        base = self.energy_score + self.text_score + self.laughter_score
        if self.llm_score is not None:
            base += self.llm_score
        return base


@dataclass
class ClipResult:
    """Итоговый клип после нарезки."""

    path: str
    start: float
    end: float
    duration: float
    score: float
    text: str = ""
    reason: str = ""
    vertical: bool = False


@dataclass
class PipelineResult:
    """Сводный результат обработки."""

    input_path: str
    duration: float
    clips: list = field(default_factory=list)