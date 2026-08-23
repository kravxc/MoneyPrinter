"""Утилиты для работы с ffmpeg/ffprobe."""

from __future__ import annotations

import json
import locale
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .models import VideoInfo


def _system_encoding() -> str:
    """Кодировка вывода внешних программ (ffmpeg) в текущей ОС.

    На Windows ffmpeg пишет в OEM-кодировке (обычно cp866), из-за чего
    русский текст превращается в «кракозябры», если декодировать как UTF-8.
    """
    enc = locale.getpreferredencoding(False)
    if enc and enc.lower().startswith("cp"):
        return enc
    if sys.platform == "win32":
        return "cp866"
    return enc or "utf-8"


def _decode(raw: bytes) -> str:
    """Декодирует вывод внешней программы без «кракозябр»."""
    if not raw:
        return ""
    enc = _system_encoding()
    try:
        return raw.decode(enc, errors="replace")
    except (LookupError, UnicodeError):
        return raw.decode("utf-8", errors="replace")


class FFmpegError(RuntimeError):
    pass


def run_with_progress(
    cmd: list,
    total: float,
    desc: str = "",
    initial: float = 0.0,
    bar=None,
    disable: bool = False,
) -> str:
    """Запускает ffmpeg с прогресс-баром по длительности.

    - Добавляет `-progress pipe:1`, читает out_time_ms из stdout.
    - Если передан готовый `bar` (tqdm), использует его — удобно для единого
      бара на всю нарезку нескольких клипов.
    - Возвращает stderr (нужен для парсинга showinfo в детекции сцен).
    """
    from tqdm import tqdm

    cmd = list(cmd) + ["-progress", "pipe:1", "-nostats"]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False
        )
    except FileNotFoundError:
        raise FFmpegError(f"Команда не найдена: {cmd[0]}. Установите ffmpeg.") from None

    own = bar is None
    if own:
        bar = tqdm(total=total, desc=desc, unit="s", disable=disable or total <= 0)

    try:
        for raw_line in proc.stdout:
            line = _decode(raw_line)
            if line.startswith("out_time_ms="):
                try:
                    secs = int(line.split("=", 1)[1].strip()) / 1_000_000
                except ValueError:
                    continue
                bar.n = initial + min(secs, max(total, 0.0))
                bar.refresh()
        leftover = proc.stdout.read()
    finally:
        if own:
            bar.close()

    stderr = _decode(proc.stderr.read())
    leftover_txt = _decode(leftover)
    proc.wait()
    if proc.returncode != 0:
        tail = (stderr or leftover_txt or "")[-2000:]
        raise FFmpegError(f"Команда завершилась с ошибкой {proc.returncode}:\n{tail}")
    return stderr


def _run(cmd: list, *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Запуск внешней команды с понятным сообщением об ошибке."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            text=False,
            check=False,
        )
    except FileNotFoundError:
        raise FFmpegError(f"Команда не найдена: {cmd[0]}. Установите ffmpeg.") from None
    stdout = _decode(result.stdout) if capture else ""
    stderr = _decode(result.stderr) if capture else ""
    if check and result.returncode != 0:
        tail = (stderr or stdout or "")[-2000:]
        raise FFmpegError(f"Команда завершилась с ошибкой {result.returncode}:\n{tail}")
    return subprocess.CompletedProcess(result.args, result.returncode, stdout, stderr)


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise FFmpegError("ffmpeg/ffprobe не найдены в PATH. Установите: brew install ffmpeg")


def probe(path: str) -> VideoInfo:
    """Читает метаданные видео через ffprobe."""
    require_ffmpeg()
    result = _run(
        [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    data = json.loads(result.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFmpegError(f"В файле {path} нет видеодорожки")

    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0.0)
    fps: float = 0.0
    avg = video.get("avg_frame_rate", "0/1")
    try:
        num, den = avg.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))

    return VideoInfo(
        path=str(path),
        duration=duration,
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        fps=fps,
        has_audio=has_audio,
    )


def decode_audio_wav(
    path: str,
    out_wav: str,
    sample_rate: int = 16000,
    mono: bool = True,
) -> str:
    """Декодирует аудио во временный WAV для анализа (16k моно)."""
    require_ffmpeg()
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(path),
        "-vn",
        "-ac", "1" if mono else "2",
        "-ar", str(sample_rate),
        "-f", "wav",
        str(out_wav),
    ]
    _run(cmd)
    return out_wav


def extract_frame(path: str, out_png: str, time: float = 0.0) -> str:
    """Извлекает один кадр в заданный момент времени."""
    require_ffmpeg()
    _run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-ss", f"{time:.3f}",
            "-i", str(path),
            "-frames:v", "1",
            "-q:v", "2",
            str(out_png),
        ]
    )
    return out_png


def duration_of(path: str) -> float:
    return probe(path).duration