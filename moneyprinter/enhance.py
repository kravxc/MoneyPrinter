"""Улучшение качества видео (апскейл, шумоподавление, резкость, AI-апскейл).

Используется как standalone-команда `moneyprinter enhance`, а также опционально
в пайплайне `serial`/`process` через флаг --enhance.

Режимы:
  * ffmpeg-only (по умолчанию) — быстрое улучшение через фильтры ffmpeg:
    шумоподавление (hqdn3d) → резкость (unsharp) → масштабирование (lanczos).
  * AI (--ai) — Real-ESRGAN через ncnn-vulkan для супер-разрешения.
    Требует установленный бинарник `realesrgan-ncnn-vulkan` в PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .media import FFmpegError, _decode, probe, require_ffmpeg


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EnhanceConfig:
    """Параметры улучшения видео."""
    target_width: int = 1920
    target_height: int = 1080
    crf: int = 18                   # качество (меньше = лучше)
    preset: str = "slow"            # скорость кодирования (медленнее = лучше)
    denoise_strength: int = 3       # сила шумоподавления hqdn3d (0 = выкл)
    sharpen_strength: float = 1.5   # сила резкости unsharp (0 = выкл)
    sharp_mode: str = "cas+unsharp" # unsharp | cas | cas+unsharp | off
    preserve_aspect: bool = True    # сохранять пропорции/ориентацию исходника
    use_ai: bool = False            # включить Real-ESRGAN
    ai_model: str = "realesrgan-x4plus"  # модель Real-ESRGAN
    ai_scale: int = 2               # коэффициент масштабирования AI (2 или 4)
    jobs: int = 0                   # параллельность (0 = все ядра)


# ---------------------------------------------------------------------------
# ffmpeg фильтры
# ---------------------------------------------------------------------------

def _resolve_scale(cfg: EnhanceConfig, src_w: int, src_h: int) -> str:
    """Возвращает scale-фильтр с учётом ориентации исходника.

    Если preserve_aspect=True (по умолчанию) и пользователь не менял формат,
    target подстраивается под ориентацию видео:
      * горизонтальное (шир > выс) → target сохраняется (обычно 1920x1080)
      * вертикальное (выс > шир)   → target меняется местами (напр. 1080x1920)
    Содержимое не искажается — используется force_original_aspect_ratio=decrease
    и pad до целевого размера (чёрные/размытые поля при несовпадении пропорций).
    """
    tw, th = cfg.target_width, cfg.target_height
    if cfg.preserve_aspect and tw and th:
        src_vertical = src_h > src_w
        tgt_vertical = th > tw
        # если ориентация исходника и цели не совпадают — меняем местами
        if src_vertical != tgt_vertical:
            tw, th = th, tw
    if not tw or not th:
        # целевой размер не задан — просто upscale в 2 раза (сохраняя пропорции)
        sf = 2
        target_scale = f"scale={src_w * sf}:{src_h * sf}:flags=lanczos"
        return target_scale
    return (
        f"scale={tw}:{th}:flags=lanczos:force_original_aspect_ratio=decrease"
        f":force_divisible_by=2,pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black"
    )


def _build_vf(cfg: EnhanceConfig, src_w: int = 0, src_h: int = 0) -> str:
    """Собирает строку -vf для ffmpeg: шумоподавление → резкость → scale.

    Резкость заметно усиливается (для последующего пересжатия соцсетями):
      * cas (Contrast Adaptive Sharpening) — самый заметный эффект,
        чёткость без ореолов (как в upscaler'ях NVIDIA);
      * unsharp — классическое усиление контуров.
    Режим cas+unsharp даёт максимально резкую картинку.
    """
    parts: list[str] = []

    # 1) шумоподавление (ДО резкости, иначе резкость усиливает шумы)
    if cfg.denoise_strength > 0:
        s = cfg.denoise_strength
        parts.append(f"hqdn3d={s}:{s}:{s-1}:{s}")

    # 2) резкость
    mode = (cfg.sharp_mode or "unsharp").lower()
    if cfg.sharpen_strength > 0 and mode != "off":
        l = max(0.0, cfg.sharpen_strength)
        if "cas" in mode:
            # cas: 0..1, ~0.5 уже заметно; для «сильной» резкости берём до 1.0
            cas_strength = min(1.0, 0.4 + l * 0.3)
            parts.append(f"cas={cas_strength:.2f}")
        if "unsharp" in mode:
            parts.append(f"unsharp=5:5:{l}:5:5:{l * 0.5}")

    # 3) масштабирование (сохраняя пропорции исходника)
    if src_w and src_h:
        parts.append(_resolve_scale(cfg, src_w, src_h))
    else:
        tw, th = cfg.target_width, cfg.target_height
        parts.append(f"scale={tw}:{th}:flags=lanczos")

    return ",".join(parts)


def _build_encode_args(cfg: EnhanceConfig) -> list:
    """Общие аргументы кодирования для ffmpeg."""
    return [
        "-c:v", "libx264",
        "-preset", cfg.preset,
        "-crf", str(cfg.crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
    ]


# ---------------------------------------------------------------------------
# Real-ESRGAN (ncnn-vulkan)
# ---------------------------------------------------------------------------

def check_realesrgan() -> bool:
    """Проверяет доступность realesrgan-ncnn-vulkan в PATH."""
    return shutil.which("realesrgan-ncnn-vulkan") is not None


def _enhance_with_ai(
    input_path: str,
    output_path: str,
    cfg: EnhanceConfig,
    src_w: int,
    src_h: int,
    total_duration: float = 0.0,
    bar=None,
) -> str:
    """Апскейл через Real-ESRGAN (ncnn-vulkan) + финальное кодирование ffmpeg.

    Шаги:
      1) realesrgan-ncnn-vulkan: input → tmp_upscaled.mp4 (апскейл AI)
      2) ffmpeg: tmp_upscaled → output (ресайз до целевого + фильтры + кодек)
    """
    if not check_realesrgan():
        raise FFmpegError(
            "realesrgan-ncnn-vulkan не найден в PATH. "
            "Установите: https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan "
            "или отключите --ai."
        )

    with tempfile.TemporaryDirectory(prefix="moneyprinter_enhance_") as tmp:
        tmp_upscaled = os.path.join(tmp, "upscaled.mp4")

        # Шаг 1: AI-апскейл
        cmd_ai = [
            "realesrgan-ncnn-vulkan",
            "-i", str(input_path),
            "-o", tmp_upscaled,
            "-n", cfg.ai_model,
            "-s", str(cfg.ai_scale),
            "-f", "mp4",
        ]
        print(f"  [AI] Real-ESRGAN апскейл x{cfg.ai_scale}...")
        try:
            result = subprocess.run(
                cmd_ai,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
            )
            if result.returncode != 0:
                stderr = _decode(result.stderr)
                raise FFmpegError(f"Real-ESRGAN завершился с ошибкой {result.returncode}:\n{stderr[-2000:]}")
        except FileNotFoundError:
            raise FFmpegError("realesrgan-ncnn-vulkan не найден. Установите или отключите --ai.")

        # Шаг 2: ffmpeg — ресайз до целевого размера + фильтры + кодек
        vf = _build_vf(cfg, src_w, src_h)
        cmd_ffmpeg = [
            "ffmpeg", "-v", "error", "-y",
            "-i", tmp_upscaled,
            "-vf", vf,
        ] + _build_encode_args(cfg) + [
            "-movflags", "+faststart",
            str(output_path),
        ]
        _run_ffmpeg_enhance(cmd_ffmpeg, total_duration, bar)

    return output_path


# ---------------------------------------------------------------------------
# Основная функция обработки
# ---------------------------------------------------------------------------

def enhance_video(
    input_path: str,
    output_path: str,
    cfg: EnhanceConfig,
    total_duration: float = 0.0,
    bar=None,
) -> str:
    """Улучшает одно видеофайл.

    Если cfg.use_ai=True — через Real-ESRGAN, иначе — ffmpeg-фильтры.
    """
    require_ffmpeg()

    # определяем исходное разрешение, чтобы сохранить ориентацию/пропорции
    try:
        info = probe(input_path)
        src_w, src_h = int(info.width), int(info.height)
    except Exception:
        src_w, src_h = 0, 0

    if cfg.use_ai:
        return _enhance_with_ai(input_path, output_path, cfg, src_w, src_h, total_duration, bar)

    # ffmpeg-only режим
    vf = _build_vf(cfg, src_w, src_h)
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-hwaccel", "auto",
        "-i", str(input_path),
        "-vf", vf,
    ] + _build_encode_args(cfg) + [
        "-movflags", "+faststart",
        str(output_path),
    ]
    _run_ffmpeg_enhance(cmd, total_duration, bar)
    return output_path


def _run_ffmpeg_enhance(cmd: list, total_duration: float = 0.0, bar=None) -> None:
    """Запускает ffmpeg с прогресс-баром."""
    own = bar is None
    if own and total_duration > 0:
        bar = tqdm(total=total_duration, desc="Улучшение", unit="s")
    elif own:
        bar = tqdm(desc="Улучшение", unit="s", disable=True)

    cmd_full = cmd + ["-progress", "pipe:1", "-nostats"]
    try:
        proc = subprocess.Popen(
            cmd_full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False
        )
    except FileNotFoundError:
        raise FFmpegError(f"Команда не найдена: {cmd_full[0]}. Установите ffmpeg.") from None

    try:
        for raw_line in proc.stdout:
            line = _decode(raw_line)
            if line.startswith("out_time_ms="):
                try:
                    secs = int(line.split("=", 1)[1].strip()) / 1_000_000
                except ValueError:
                    continue
                if bar and total_duration > 0:
                    bar.n = min(secs, total_duration)
                    bar.refresh()
        proc.stdout.read()
    finally:
        if own and bar:
            bar.close()

    stderr = _decode(proc.stderr.read())
    proc.wait()
    if proc.returncode != 0:
        tail = stderr[-2000:] if stderr else ""
        raise FFmpegError(f"ffmpeg завершился с ошибкой {proc.returncode}:\n{tail}")


# ---------------------------------------------------------------------------
# Пакетная обработка
# ---------------------------------------------------------------------------

def enhance_directory(
    input_dir: str,
    output_dir: str,
    cfg: EnhanceConfig,
) -> List[str]:
    """Улучшает все .mp4/.mov/.mkv в папке параллельно.

    Возвращает список обработанных файлов.
    """
    require_ffmpeg()
    exts = ("*.mp4", "*.mov", "*.mkv", "*.webm")
    files = []
    for ext in exts:
        files.extend(Path(input_dir).glob(ext))
    files = sorted(files)

    if not files:
        print("[i] Видеофайлы не найдены.")
        return []

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    jobs = cfg.jobs or os.cpu_count() or 1
    results: List[str] = []
    errors: List[str] = []

    print(f"[i] Улучшение {len(files)} файлов ({jobs} потоков)...")

    def _process_one(f: Path) -> str:
        out_name = f.stem + "_enhanced.mp4"
        out_path = str(out / out_name)
        info = probe(str(f))
        enhance_video(str(f), out_path, cfg, total_duration=info.duration)
        return out_path

    bar = tqdm(total=len(files), desc="Файлы", unit="файл")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_process_one, f): f for f in files}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                path = fut.result()
                results.append(path)
                bar.update(1)
                bar.set_postfix_str(Path(path).name)
            except Exception as exc:
                errors.append(f"{f.name}: {exc}")
                bar.update(1)
    bar.close()

    if errors:
        print(f"\n[warn] Ошибки ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
    print(f"[✓] Улучшено: {len(results)} файлов → {output_dir}")
    return results
