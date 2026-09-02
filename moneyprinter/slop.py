"""Генерация «нейрослоп»-видео: абсурдная история → озвучка → клип.

Воссоздаёт вирусный жанр коротких AI-видео (типа «я — клубника, ты —
клубника, почему у нас родился банан»). Для генерации не нужен GPU и модели:
  * история (промпт) — ваш текст, файл или эвристически сгенерированная;
  * озвучка — локальный TTS (say на macOS, espeak/piper на других);
  * кадры — процедурные «AI-лоп» фоны (градиенты/шум) + крупные титры,
    создаются через ffmpeg;
  * итог — вертикальный 1080x1920 клип для Shorts/TikTok.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .media import require_ffmpeg

TARGET_W, TARGET_H = 1080, 1920

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SlopConfig:
    story: str = ""                 # текст истории (если задано напрямую)
    story_file: str = ""            # файл с текстом истории
    output: str = "slop"            # папка для результата
    background: str = "gradient"    # gradient | glitch
    title_spacing: float = 0.35     # доля времени сцены на титр (0..1)
    voice: str = ""                 # голос TTS (пусто = авто: Milena на macOS)
    fps: int = 24
    rate: int = 160                 # скорость речи (135-200), работает для espeak/piper
    seed: int = 0                   # 0 = случайный
    tiktok_style: bool = False      # добавить кроп-титры (как в TikTok-нейрослопах)


# ---------------------------------------------------------------------------
# История (промпт)
# ---------------------------------------------------------------------------

_ABSTRACT_THEMES = [
    ("клубника", "банан"),
    ("пицца", "ананас"),
    ("робот", "кофе"),
    ("огурчик", "капуста"),
    ("кот", "акула"),
    ("лужайка", "трактор"),
]

_SENTENCE_TEMPLATES = [
    "я — {a}, ты — {a}",
    "почему у нас родился {b}",
    "мы шли по дороге и увидели {b}",
    "это {a}, это {a}, это всё {a}",
    "а {b} просто смотрел на нас",
    "мы спросили у {b}, зачем ты здесь",
    "{b} ничего не ответил и ушёл",
    "потом появился {b} и мы стали друзьями",
    "теперь мы все вместе — {a} и {b}",
    "спросите меня почему, но ответа нет",
    "это просто жизнь, — сказал {b}",
    "и мы поняли, что всё это имеет смысл",
]


def _random_abject_story(seed: int = 0) -> str:
    """Генерирует абсурдную историю в духе «я — клубника, ты — клубника…»."""
    rnd = random.Random(seed)
    a, b = rnd.choice(_ABSTRACT_THEMES)
    n = rnd.randint(6, 9)
    sentences = [rnd.choice(_SENTENCE_TEMPLATES).format(a=a, b=b) for _ in range(n)]
    # маленький крючок-заголовок
    head = f"я {a}, ты {a}"
    body = " ".join(sentences).capitalize()
    return f"{head}. {body}."


def resolve_story(cfg: SlopConfig) -> str:
    """Возвращает текст истории из cfg (прямой текст > файл > случайная)."""
    story = (cfg.story or "").strip()
    if not story and cfg.story_file:
        p = Path(cfg.story_file)
        if p.exists():
            story = p.read_text(encoding="utf-8").strip()
        else:
            raise FileNotFoundError(f"файл истории не найден: {cfg.story_file}")
    if not story:
        story = _random_abject_story(cfg.seed)
        print(f"[i] История не задана — сгенерирована случайная:\n   «{story[:120]}…»")
    return story


def split_scenes(story: str) -> List[str]:
    """Делит историю на сцены по предложениям (не больше ~12 сцен)."""
    import re
    parts = re.split(r"(?<=[.!?…])\s+", story.strip().replace("\n", " "))
    scenes = [p.strip() for p in parts if p.strip()]
    return scenes[:12]


# ---------------------------------------------------------------------------
# TTS (локальная озвучка)
# ---------------------------------------------------------------------------

def _detect_tts() -> str:
    """Выбирает доступный TTS: piper > say (macOS) > espeak-ng/espeak > Windows SAPI."""
    if shutil.which("piper"):
        return "piper"
    import sys
    if sys.platform == "darwin" and shutil.which("say"):
        return "say"
    for name in ("espeak-ng", "espeak"):
        if shutil.which(name):
            return name
    if sys.platform == "win32":
        # встроенный в Windows SAPI через PowerShell — работает без установок
        return "powershell"
    return ""


def _tts(text: str, out_wav: str, cfg: SlopConfig) -> str:
    """Озвучивает текст через локальный TTS. Возвращает путь к аудиофайлу."""
    engine = _detect_tts()
    if not engine:
        raise RuntimeError(
            "Не найден локальный TTS (piper/say/espeak). "
            "На macOS он встроен (say); на Linux установите espeak-ng."
        )

    if engine == "say":
        # say пишет только AIFF (WAV даёт ошибку fmt?). ffmpeg читает aiff.
        aiff = out_wav.rsplit(".", 1)[0] + ".aiff"
        cmd = ["say", "-o", aiff]
        voice = _detect_voice(cfg)
        if voice:
            cmd += ["-v", voice]
        cmd.append(text)
        subprocess.run(cmd, check=True)
        return aiff
    elif engine == "piper":
        if not cfg.voice:
            raise RuntimeError("Для piper укажите путь к русской модели через --voice.")
        with open(out_wav, "wb") as fh:
            subprocess.run(["piper", "-f", text, "-m", cfg.voice], stdout=fh, check=True)
        return out_wav
    elif engine == "powershell":
        return _tts_powershell(text, out_wav)
    else:  # espeak / espeak-ng
        cmd = [engine]
        if _ru_needed(text):
            cmd += ["-v", "ru"]
        cmd += ["-s", str(cfg.rate), "-w", out_wav, text]
        subprocess.run(cmd, check=True)
        return out_wav


def _tts_powershell(text: str, out_wav: str) -> str:
    """Windows SAPI (встроенный) через PowerShell → WAV."""
    ps_script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.SetOutputToWaveFile('{out}'); "
        "$s.Speak('{text}'); $s.Dispose()"
    ).format(out=out_wav.replace("'", "''"), text=text.replace("'", "''"))
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script], check=True
    )
    return out_wav


def _ru_needed(text: str) -> bool:
    return any("\u0400" <= ch <= "\u04FF" for ch in text)


def _say_has_ru() -> bool:
    try:
        out = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True
        ).stdout or ""
        return "ru_RU" in out or "Milena" in out
    except Exception:
        return False


def _detect_voice(cfg: SlopConfig) -> str:
    if cfg.voice:
        return cfg.voice
    if _detect_tts() == "say" and _say_has_ru():
        return "Milena"
    return ""


# ---------------------------------------------------------------------------
# Процедурные фоны (ffmpeg)
# ---------------------------------------------------------------------------

def _background_cmd(cfg: SlopConfig, seconds: float, seed: int) -> list:
    """Возвращает ffmpeg-команду, генерирующую фоновый клип длительностью seconds."""
    out = f"bg_{seed}.mp4"
    if cfg.background == "glitch":
        # «AI-телевизор»: градиенты + сильное вращение оттенка + лёгкий шум.
        # Не используем чистый шум (он не сжимается и даёт мегабайтный файл).
        vf = (
            f"gradients=s=1080x1920:d={seconds:.3f}:r={cfg.fps}:"
            f"nb_colors=3:speed=0.25:type=radial:seed={seed},"
            f"hue=H={seed}:s=2.2,eq=saturation=2.0:contrast=1.4,"
            f"noise=alls=10:allf=t"
        )
    else:
        # психоделический градиентный фон
        vf = (
            f"gradients=s=1080x1920:d={seconds:.3f}:r={cfg.fps}:"
            f"nb_colors=4:speed=0.1:type=radial:seed={seed},"
            f"hue=H={seed}:s=1.5,eq=saturation=1.6:contrast=1.15,"
            f"noise=alls=6:allf=t"
        )
    return ["ffmpeg", "-v", "error", "-y", "-filter_complex", vf, "-frames:v", str(int(seconds * cfg.fps)), "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", out]


def _title_cmd(scene: str, seconds: float, fps: int, fontfile: str, seed: int) -> list:
    """Генерирует сцену с крупным цветным титром поверх фона.

    Текст передаётся через textfile (файл) — так не нужно экранировать
    кавычки/двоеточия в drawtext, что критично на Windows. Пути пишем со
    слешами `/` (ffmpeg принимает их и на Windows).
    """
    colors = ["#FF2D55", "#00C2FF", "#FFEB3B", "#7CFF00", "#FF5C00"]
    color = colors[seed % len(colors)]
    # textfile: utf-8 без BOM (drawtext требует)
    textfile = f"caption_{seed}.txt"
    vf = (
        f"drawtext=fontfile={_ffpath(fontfile)}:"
        f"textfile={textfile}:"
        f"fontcolor={color}:fontsize=110:line_spacing=30:"
        f"x=(w-text_w)/2:y=(h-text_h)/2:"
        f"shadowcolor=black:shadowx=6:shadowy=6:"
        f"box=1:boxcolor=black@0.55:boxborderw=40"
    )
    # берём уже сгенерированный фон, накладываем титр
    return ["ffmpeg", "-v", "error", "-y", "-i", f"bg_{seed}.mp4", "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", f"title_{seed}.mp4"], textfile, scene


def _ffpath(path: str) -> str:
    """Нормализует путь для ffmpeg-фильтра: всегда слэши, экранирование ' и %."""
    p = path.replace("\\", "/")
    p = p.replace(":", "\\:").replace("'", "\\'").replace("%", "%%")
    return p


def write_utf8(path: Path, text: str) -> None:
    """Пишет UTF-8 БЕЗ BOM (янв. для текстовых файлов drawtext)."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# Сборка
# ---------------------------------------------------------------------------

def _probe_duration(audio_or_video: str) -> float:
    """Возвращает длительность файла (аудио или видео) по ffprobe."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_or_video)],
            capture_output=True, text=True,
        ).stdout.strip()
        return float(out) if out else 1.0
    except Exception:
        return 1.0


def generate_slop(cfg: SlopConfig) -> str:
    """Генерирует «нейрослоп»-видео. Возвращает путь к итоговому клипу."""
    require_ffmpeg()
    story = resolve_story(cfg)
    scenes = split_scenes(story)
    if not scenes:
        raise ValueError("История пуста — нечего озвучивать.")

    out_dir = Path(cfg.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.seed or (random.SystemRandom().randint(1, 9999))

    with tempfile.TemporaryDirectory(prefix="slop_") as tmp:
        tmp_path = Path(tmp)

        # озвучиваем каждую сцену
        wav_files: List[Path] = []
        print(f"[i] Озвучка {len(scenes)} сцен ({_detect_tts()})...")
        for i, sc in enumerate(scenes, 1):
            wav = tmp_path / f"wav_{i}.wav"
            created = _tts(sc, str(wav), cfg)
            # нормализуем любой формат TTS (aiff/wav/...) в единый PCM WAV —
            # надёжно для конкатенации на любой ОС
            norm = tmp_path / f"norm_{i}.wav"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", created,
                 "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(norm)],
                check=True,
            )
            wav_files.append(norm)

        # генерируем кадр/фон для каждой сцены и титр
        scene_clips: List[str] = []
        fonts = _find_font()
        fontfile = None
        if fonts:
            # копируем шрифт в локальный файл с простым именем — чтобы в
            # drawtext не передавать абсолютный путь с двоеточиями/слэшами
            # (это ломает парсер фильтра на Windows). fontfile=font.ttf
            # относительно cwd работает везде.
            try:
                shutil.copy(fonts[0], tmp_path / "font.ttf")
                fontfile = "font.ttf"
            except OSError:
                fontfile = None
        for i, (sc, wav) in enumerate(zip(scenes, wav_files), 1):
            dur = _probe_duration(str(wav)) * 1.15
            if dur < 1.0:
                dur = 1.0
            subprocess.run(_background_cmd(cfg, dur, seed + i), cwd=str(tmp_path), check=True)
            if fontfile:
                cmd, textfile, _scene = _title_cmd(sc, dur, cfg.fps, fontfile, seed + i)
                # пишем текст титра в файл (utf-8 без BOM — требование drawtext)
                write_utf8(tmp_path / textfile, _scene)
                subprocess.run(cmd, cwd=str(tmp_path), check=True)
                scene_clips.append(f"title_{seed + i}.mp4")
            else:
                scene_clips.append(f"bg_{seed + i}.mp4")

        # склейка: видео + аудио
        final = out_dir / f"slop_{seed}.mp4"
        _concat(tmp_path, scene_clips, wav_files, final, cfg)
        print(f"[✓] Готово: {final}")

    return str(final)


def _concat(tmp: Path, scene_clips: List[str], wavs: List[Path], out: Path, cfg: SlopConfig) -> None:
    """Собирает итоговый клип из сцен + голоса."""
    # сначала склеиваем видео
    listfile = tmp / "list.txt"
    with open(listfile, "w", encoding="utf-8") as fh:
        for c in scene_clips:
            fh.write(f"file '{c}'\n")
    concat_video = tmp / "concat_video.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
         "-c", "copy", str(concat_video)],
        cwd=str(tmp), check=True,
    )

    # склеиваем аудио (голоса) в одну дорожку через concat list (кроссплатформенно)
    audiolist = tmp / "audio_list.txt"
    with open(audiolist, "w", encoding="utf-8") as fh:
        for w in wavs:
            fh.write(f"file '{w}'\n")
    concat_audio = tmp / "concat_audio.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(audiolist),
         "-c", "copy", str(concat_audio)],
        cwd=str(tmp), check=False,
    )

    # накладываем аудио на видео
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(concat_video), "-i", str(concat_audio),
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k",
         "-shortest", "-movflags", "+faststart", str(out)],
        check=True,
    )


def _find_font() -> List[str]:
    """Ищет доступный шрифт для drawtext (ttf/ttc/otf)."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    found = [p for p in candidates if os.path.exists(p) and (p.endswith((".ttf", ".ttc", ".otf")))]
    if not found:
        import glob as _g
        for pat in ("/System/Library/Fonts/*.ttc", "/System/Library/Fonts/*.otf",
                    "/usr/share/fonts/**/*.ttf"):
            found.extend(_g.glob(pat))
    return found
