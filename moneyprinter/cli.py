"""CLI интерфейс MoneyPrinter."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from .pipeline import Config, process
from .media import probe
from . import upload as upload_mod
from . import scheduler as scheduler_mod
from . import hashtags as hashtags_mod


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moneyprinter",
        description="Режет стримы/сериалы/фильмы на яркие вертикальные клипы для Shorts/TikTok.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("process", help="Обработать видео и нарезать клипы")
    p_run.add_argument("input", help="Путь к видеофайлу")
    p_run.add_argument("-o", "--output", default="clips", help="Папка для клипов (default: clips)")
    p_run.add_argument("-n", "--max-clips", type=int, default=10, help="Сколько клипов сделать (default: 10)")
    p_run.add_argument("--min-duration", type=float, default=60.0, help="Мин. длина клипа, сек (default: 60 — минимум для сериалов/фильмов)")
    p_run.add_argument("--max-duration", type=float, default=180.0, help="Макс. длина клипа, сек (Shorts допускает до 180)")
    p_run.add_argument("--story-gap", type=float, default=2.0, help="Макс. пауза между репликами одной мысли, сек")
    p_run.add_argument("--min-score", type=float, default=0.0, help="Отсечка по суммарному скору")
    p_run.add_argument("--horizontal", action="store_true", help="Не конвертировать в 9:16 (оставить пропорции)")
    p_run.add_argument("--crop", action="store_true", help="Вертикаль кадрированием вместо размытого фона")
    p_run.add_argument("--whisper-model", default="base", help="Модель Whisper: tiny/base/small/medium/large")
    p_run.add_argument("--device", default="auto", help="auto/cpu/cuda (auto = cpu; cuda ускорит на NVIDIA GPU)")
    p_run.add_argument("--jobs", type=int, default=0, help="Сколько клипов нарезать параллельно (0 = все ядра)")
    p_run.add_argument("--no-auto-install", action="store_true", help="Не доустанавливать AI-зависимости автоматически")
    p_run.add_argument("--language", default=None, help="Язык для транскрипции (напр. ru, en)")
    p_run.add_argument("--llm", default=None, metavar="MODEL", help="Локальная LLM для ранжирования и хештегов (напр. llama3.2)")
    p_run.add_argument("--llm-url", default=None, help="Ollama base url (default: http://localhost:11434)")
    p_run.add_argument("--scene-threshold", type=float, default=27.0, help="Чувствительность детекции сцен")
    p_run.add_argument("--schedule", action="store_true", help="Сразу поставить нарезанные клипы в очередь публикации (первый сразу, остальные по интервалу)")
    p_run.add_argument("--schedule-interval", type=float, default=7200.0, help="Интервал между отложенными публикациями, сек (default: 7200 = 2ч)")
    p_run.add_argument("--no-upload", action="store_true", help="Не запускать планировщик после нарезки (только сформировать очередь)")

    p_probe = sub.add_parser("probe", help="Показать метаданные видео")
    p_probe.add_argument("input", help="Путь к видеофайлу")

    p_login = sub.add_parser("login", help="Войти в TikTok и сохранить cookies для автопостинга")
    p_login.add_argument("--cookie", default=upload_mod.DEFAULT_COOKIE_FILE, help="Куда сохранить cookies")
    p_login.add_argument("--headed", action="store_true", help="Открыть браузер видимым (нужно для ручного входа)")

    p_publish = sub.add_parser("publish", help="Выложить видео в TikTok (один файл или папку)")
    p_publish.add_argument("path", help="Видеофайл или папка с клипами")
    p_publish.add_argument("--caption", default="", help="Подпись/хештеги (иначе сгенерируются по имени/транскрипции)")
    p_publish.add_argument("--cookie", default=upload_mod.DEFAULT_COOKIE_FILE, help="Файл cookies TikTok")
    p_publish.add_argument("--headed", action="store_true", help="Показывать браузер")
    p_publish.add_argument("--queue", action="store_true", help="Не публиковать сразу, а добавить в очередь планировщика")
    p_publish.add_argument("--schedule-interval", type=float, default=7200.0, help="Интервал между публикациями, сек (default: 7200 = 2ч)")
    p_publish.add_argument("--state", default=scheduler_mod.DEFAULT_STATE_FILE, help="Файл состояния очереди")

    p_sched = sub.add_parser("run-schedule", help="Держать очередь и публиковать по расписанию")
    p_sched.add_argument("--state", default=scheduler_mod.DEFAULT_STATE_FILE, help="Файл состояния очереди")
    p_sched.add_argument("--schedule-interval", type=float, default=7200.0, help="Интервал между публикациями, сек (default: 7200 = 2ч)")
    p_sched.add_argument("--cookie", default=upload_mod.DEFAULT_COOKIE_FILE, help="Файл cookies TikTok")
    p_sched.add_argument("--dry-run", action="store_true", help="Не публиковать реально, только показывать план")

    return parser


def _cmd_probe(args: argparse.Namespace) -> int:
    info = probe(args.input)
    print(f"path:     {info.path}")
    print(f"duration: {info.duration:.2f}s")
    print(f"size:     {info.width}x{info.height}")
    print(f"fps:      {info.fps:.2f}")
    print(f"audio:    {'yes' if info.has_audio else 'no'}")
    return 0


def _collect_videos(path: str) -> list:
    if os.path.isdir(path):
        exts = ("*.mp4", "*.mov", "*.mkv", "*.webm")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(path, ext)))
        return sorted(files)
    return [path]


def _cmd_login(args: argparse.Namespace) -> int:
    upload_mod.interactive_login(path=args.cookie, headed=args.headed or True)
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    videos = _collect_videos(args.path)
    captions = []
    for v in videos:
        if args.caption:
            captions.append(args.caption)
        else:
            # фолбэк: базовые теги по имени файла
            tags = hashtags_mod.generate_hashtags(os.path.splitext(os.path.basename(v))[0])
            captions.append(hashtags_mod.build_caption("", tags))
    if args.queue:
        scheduler_mod.plan_queue(
            videos, captions, interval=args.schedule_interval, state_file=args.state
        )
        print(f"[✓] Добавлено в очередь ({len(videos)}): {args.state}")
        print(f"    Запустите `moneyprinter run-schedule` для публикации (интервал {args.schedule_interval/3600:.1f}ч).")
    else:
        for v, cap in zip(videos, captions):
            upload_mod.upload_video(v, cap, cookie_path=args.cookie, headed=args.headed)
    return 0


def _cmd_run_schedule(args: argparse.Namespace) -> int:
    scheduler_mod.run_schedule(
        interval=args.schedule_interval,
        state_file=args.state,
        cookie_path=args.cookie,
        dry_run=args.dry_run,
    )
    return 0


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "probe":
            return _cmd_probe(args)
        if args.command == "login":
            return _cmd_login(args)
        if args.command == "publish":
            return _cmd_publish(args)
        if args.command == "run-schedule":
            return _cmd_run_schedule(args)

        cfg = Config(
            input_path=args.input,
            output_dir=args.output,
            max_clips=args.max_clips,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            min_score=args.min_score,
            vertical=not args.horizontal,
            blur_bg=not args.crop,
            story_gap=args.story_gap,
            whisper_model=args.whisper_model,
            device=args.device,
            language=args.language,
            scene_threshold=args.scene_threshold,
            llm_model=args.llm,
            llm_url=args.llm_url,
            jobs=args.jobs,
            auto_install=not args.no_auto_install,
        )
        result = process(cfg)
        print(f"\nГотово: {len(result.clips)} клипов → {args.output}")

        if args.schedule:
            videos = [c.path for c in result.clips]
            captions = [c.caption for c in result.clips]
            scheduler_mod.plan_queue(
                videos, captions, interval=args.schedule_interval, state_file=scheduler_mod.DEFAULT_STATE_FILE
            )
            print(f"[✓] Клипы поставлены в очередь публикации (интервал {args.schedule_interval/3600:.1f}ч).")
            if not args.no_upload:
                print("[i] Запускаю планировщик (Ctrl+C — сохранить состояние и выйти)...")
                scheduler_mod.run_schedule(
                    interval=args.schedule_interval,
                    state_file=scheduler_mod.DEFAULT_STATE_FILE,
                    cookie_path=upload_mod.DEFAULT_COOKIE_FILE,
                )
        return 0
    except Exception as exc:  # noqa: BLE001 — CLI должен показать дружелюбную ошибку
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())