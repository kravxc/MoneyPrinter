"""CLI интерфейс MoneyPrinter."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

from .pipeline import Config, process
from .media import probe
from . import upload as upload_mod
from . import scheduler as scheduler_mod
from . import hashtags as hashtags_mod
from . import serial as serial_mod
from . import enhance as enhance_mod


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moneyprinter",
        description="Режет стримы/сериалы/фильмы на яркие вертикальные клипы для Shorts/TikTok.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("process", help="Обработать видео и нарезать клипы")
    p_run.add_argument("input", help="Путь к видеофайлу")
    p_run.add_argument("-o", "--output", default="cuts", help="Папка для клипов (default: clips)")
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
    p_run.add_argument("--enhance", action="store_true", help="Улучшить качество клипов после нарезки (апскейл + резкость)")

    p_probe = sub.add_parser("probe", help="Показать метаданные видео")
    p_probe.add_argument("input", help="Путь к видеофайлу")

    p_login = sub.add_parser("login", help="Войти в TikTok и сохранить cookies для автопостинга")
    p_login.add_argument("--cookie", default=upload_mod.DEFAULT_COOKIE_FILE, help="Куда сохранить cookies")
    p_login.add_argument("--profile", default=upload_mod.DEFAULT_PROFILE_DIR, help="Каталог persistent-профиля браузера (для входа через Google)")
    p_login.add_argument("--force-new", action="store_true", help="Стереть старый профиль и войти заново")
    p_login.add_argument("--no-chrome", action="store_true", help="Не использовать реальный Chrome (взять Playwright Chromium)")

    p_publish = sub.add_parser("publish", help="Выложить видео в TikTok (один файл или папку)")
    p_publish.add_argument("path", help="Видеофайл или папка с клипами")
    p_publish.add_argument("--caption", default="", help="Подпись/хештеги (иначе сгенерируются по имени/транскрипции)")
    p_publish.add_argument("--cookie", default=upload_mod.DEFAULT_COOKIE_FILE, help="Файл cookies TikTok")
    p_publish.add_argument("--headed", action="store_true", help="Показывать браузер")
    p_publish.add_argument("--queue", action="store_true", help="Не публиковать сразу, а добавить в очередь планировщика")
    p_publish.add_argument("--schedule-interval", type=float, default=7200.0, help="Интервал между публикациями, сек (default: 7200 = 2ч)")
    p_publish.add_argument("--state", default=scheduler_mod.DEFAULT_STATE_FILE, help="Файл состояния очереди")

    p_sched = sub.add_parser("run-schedule", help="Держать очередь и публиковать по расписанию (долгий процесс)")
    p_sched.add_argument("--state", default=scheduler_mod.DEFAULT_STATE_FILE, help="Файл состояния очереди")
    p_sched.add_argument("--schedule-interval", type=float, default=7200.0, help="Интервал между публикациями, сек (default: 7200 = 2ч)")
    p_sched.add_argument("--cookie", default=upload_mod.DEFAULT_COOKIE_FILE, help="Файл cookies TikTok")
    p_sched.add_argument("--dry-run", action="store_true", help="Не публиковать реально, только показывать план")

    p_next = sub.add_parser("publish-next", help="Опубликовать один наступивший клип и выйти (для cron/launchd/schtasks)")
    p_next.add_argument("--state", default=scheduler_mod.DEFAULT_STATE_FILE, help="Файл состояния очереди")
    p_next.add_argument("--schedule-interval", type=float, default=7200.0, help="Интервал, сек (используется при повторе ошибки)")
    p_next.add_argument("--cookie", default=upload_mod.DEFAULT_COOKIE_FILE, help="Файл cookies TikTok")
    p_next.add_argument("--dry-run", action="store_true", help="Не публиковать реально")

    p_inst = sub.add_parser("install-scheduler", help="Поставить автозапуск publish-next в системный планировщик (cron/launchd/schtasks)")
    p_inst.add_argument("--schedule-interval", type=float, default=7200.0, help="Интервал просыпания, сек (default: 7200 = 2ч)")
    p_inst.add_argument("--state", default=scheduler_mod.DEFAULT_STATE_FILE, help="Файл состояния очереди")
    p_inst.add_argument("--cookie", default=upload_mod.DEFAULT_COOKIE_FILE, help="Файл cookies TikTok")
    p_inst.add_argument("--dry-run", action="store_true", help="Только показать, что будет создано")

    p_uninst = sub.add_parser("uninstall-scheduler", help="Удалить автозапуск из системного планировщика")
    p_uninst.add_argument("--dry-run", action="store_true", help="Только показать, что будет удалено")

    _add_serial_parser(sub)

    p_regen = sub.add_parser("regen-captions", help="Перегенерировать описания/хештеги для уже нарезанных клипов (по report.json)")
    p_regen.add_argument("input", help="Путь к исходному видеофайлу (сериал/эпизод)")
    p_regen.add_argument("-o", "--output", default="cuts", help="Папка с клипами и report.json")
    p_regen.add_argument("--series-title", default="", help="Название сериала")
    p_regen.add_argument("--episode", type=int, default=1, help="Номер серии")
    p_regen.add_argument("--base-hashtag", action="append", default=[], help="Доп. тег (можно несколько)")
    p_regen.add_argument("--global-hashtag", action="append", default=[], help="Глобальный тег сериала (одинаковый для ВСЕХ частей/серий)")
    p_regen.add_argument("--whisper-model", default="base", help="Модель Whisper")
    p_regen.add_argument("--device", default="auto", help="auto/cpu/cuda")
    p_regen.add_argument("--language", default=None, help="Язык транскрипции")
    p_regen.add_argument("--llm", default=None, metavar="MODEL", help="Локальная LLM для описания")
    p_regen.add_argument("--llm-url", default=None, help="Ollama base url")
    p_regen.add_argument("--jobs", type=int, default=0, help="Параллельность транскрипции (0 = все ядра)")

    p_enhance = sub.add_parser("enhance", help="Улучшить качество видео (апскейл, резкость, шумоподавление)")
    p_enhance.add_argument("input", help="Видеофайл или папка с видео")
    p_enhance.add_argument("-o", "--output", default="enhanced", help="Папка для результатов (default: enhanced/)")
    p_enhance.add_argument("--target-width", type=int, default=1920, help="Целевая ширина (default: 1920)")
    p_enhance.add_argument("--target-height", type=int, default=1080, help="Целевая высота (default: 1080)")
    p_enhance.add_argument("--crf", type=int, default=18, help="Качество кодирования (меньше = лучше, default: 18)")
    p_enhance.add_argument("--preset", default="slow", help="Пreset кодирования (default: slow)")
    p_enhance.add_argument("--denoise-strength", type=int, default=3, help="Сила шумоподавления hqdn3d (0=выкл, default: 3)")
    p_enhance.add_argument("--sharpen-strength", type=float, default=1.5, help="Сила резкости (0=выкл, default: 1.5)")
    p_enhance.add_argument("--sharp-mode", dest="sharp_mode", default="cas+unsharp", help="Режим резкости: cas+unsharp (макс, default) | cas | unsharp | off")
    p_enhance.add_argument("--no-preserve-aspect", action="store_true", help="Не сохранять ориентацию исходника (принудительный target формат)")
    p_enhance.add_argument("--ai", action="store_true", help="Использовать Real-ESRGAN (GPU, требует ncnn-vulkan)")
    p_enhance.add_argument("--ai-model", default="realesrgan-x4plus", help="Модель Real-ESRGAN (default: realesrgan-x4plus)")
    p_enhance.add_argument("--jobs", type=int, default=0, help="Параллельность (0 = все ядра)")

    return parser


def _add_serial_parser(sub) -> None:
    p = sub.add_parser("serial", help="Режим сериал/фильм: нарезать видео подряд по порядку на микро-серии")
    p.add_argument("input", help="Путь к видеофайлу (одна серия/эпизод)")
    p.add_argument("-o", "--output", default="cuts", help="Папка для клипов (default: clips)")
    p.add_argument("--part-duration", type=float, default=67.5, help="Средняя длина микро-серии, сек (default: 67.5 ≈ 65-70)")
    p.add_argument("--part-min", type=float, default=65.0, help="Мин. длина части, сек (default: 65)")
    p.add_argument("--part-max", type=float, default=70.0, help="Макс. длина части, сек (default: 70)")
    p.add_argument("--max-parts", type=int, default=0, help="Макс. число частей (0 = все до конца)")
    p.add_argument("--start", type=float, default=0.0, help="С какой секунды резать (default: 0)")
    p.add_argument("--end", type=float, default=0.0, help="До какой секунды (0 = до конца)")
    p.add_argument("--series-title", default="", help="Название сериала (для подписи/тегов)")
    p.add_argument("--episode", type=int, default=1, help="Номер серии (default: 1)")
    p.add_argument("--base-hashtag", action="append", default=[], help="Доп. тег (можно несколько), напр. --base-hashtag урокихимии")
    p.add_argument("--global-hashtag", action="append", default=[], help="Глобальный тег сериала (одинаковый для ВСЕХ частей/серий), напр. --global-hashtag шекер")
    p.add_argument("--horizontal", action="store_true", help="Не конвертировать в 9:16")
    p.add_argument("--crop", action="store_true", help="Вертикаль кадрированием вместо размытого фона")
    p.add_argument("--jobs", type=int, default=0, help="Сколько частей нарезать параллельно (0 = все ядра)")
    p.add_argument("--whisper-model", default="base", help="Модель Whisper для описания по содержанию (tiny/base/small/…)")
    p.add_argument("--device", default="auto", help="auto/cpu/cuda")
    p.add_argument("--language", default=None, help="Язык транскрипции (напр. ru)")
    p.add_argument("--llm", default=None, metavar="MODEL", help="Локальная LLM для описания по содержанию (напр. llama3.2)")
    p.add_argument("--llm-url", default=None, help="Ollama base url")
    p.add_argument("--no-audio-desc", action="store_true", help="Не транскрибировать (описания без крючка по содержанию)")
    p.add_argument("--schedule", action="store_true", help="Сразу поставить части в очередь публикации (первая сразу, остальные по интервалу)")
    p.add_argument("--schedule-interval", type=float, default=7200.0, help="Интервал между публикациями, сек (default: 7200 = 2ч)")
    p.add_argument("--no-upload", action="store_true", help="Не запускать планировщик после нарезки")
    p.add_argument("--enhance", action="store_true", help="Улучшить качество клипов после нарезки (апскейл + резкость)")


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
    upload_mod.interactive_login(
        path=args.cookie, profile_dir=args.profile, force_new=args.force_new,
        prefer_chrome=not args.no_chrome,
    )
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


def _cmd_publish_next(args: argparse.Namespace) -> int:
    scheduler_mod.publish_next(
        interval=args.schedule_interval,
        state_file=args.state,
        cookie_path=args.cookie,
        dry_run=args.dry_run,
    )
    return 0


def _cmd_install_scheduler(args):
    msg = scheduler_mod.install_scheduler(
        interval=args.schedule_interval,
        state_file=args.state,
        cookie_path=args.cookie,
        dry_run=args.dry_run,
    )
    print(msg)
    return 0


def _cmd_uninstall_scheduler(args):
    import shutil
    import subprocess

    if sys.platform == "win32":
        task = "MoneyPrinterTikTok"
        if args.dry_run:
            print(f"(dry-run) удалю задачу Windows «{task}»")
        else:
            subprocess.run(f'schtasks /Delete /TN "{task}" /F', shell=True, check=False)
            print(f"Удалена задача Windows «{task}».")
    elif sys.platform == "darwin":
        label = "com.moneyprinter.tiktok"
        plist = os.path.expanduser(f"~/Library/LaunchAgents/{label}.plist")
        if args.dry_run:
            print(f"(dry-run) выгружу и удалю {plist}")
        else:
            subprocess.run(["launchctl", "unload", plist], check=False)
            if os.path.exists(plist):
                os.remove(plist)
            print(f"launchd agent «{label}» удалён.")
    else:
        if args.dry_run:
            print("(dry-run) уберу строки moneyprinter-tiktok из crontab")
        else:
            out = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False).stdout or ""
            kept = [l for l in out.splitlines() if "moneyprinter-tiktok" not in l]
            subprocess.run(["crontab", "-"], input="\n".join(kept) + "\n", text=True, check=False)
            print("Строки moneyprinter-tiktok удалены из crontab.")
    return 0


def _cmd_serial(args: argparse.Namespace) -> int:
    cfg = serial_mod.SerialConfig(
        input_path=args.input,
        output_dir=args.output,
        part_duration=args.part_duration,
        part_duration_min=args.part_min,
        part_duration_max=args.part_max,
        max_parts=args.max_parts,
        start=args.start,
        end=args.end,
        vertical=not args.horizontal,
        blur_bg=not args.crop,
        series_title=args.series_title,
        episode=args.episode,
        base_hashtags=args.base_hashtag or None,
        global_hashtags=args.global_hashtag or None,
        jobs=args.jobs,
        whisper_model=args.whisper_model,
        device=args.device,
        language=args.language,
        llm_model=args.llm,
        llm_url=args.llm_url,
        auto_install=True,
        transcribe_audio=not args.no_audio_desc,
        enhance=args.enhance,
        enhance_cfg=_build_enhance_cfg(args) if getattr(args, "enhance", False) else None,
    )
    result = serial_mod.process_serial(cfg)
    print(f"\nГотово: {len(result.clips)} микро-серий → {args.output}")
    if not result.clips:
        return 0

    if args.schedule:
        videos = [c.path for c in result.clips]
        captions = [c.caption for c in result.clips]
        scheduler_mod.plan_queue(
            videos, captions, interval=args.schedule_interval, state_file=scheduler_mod.DEFAULT_STATE_FILE
        )
        print(f"[✓] Части поставлены в очередь публикации (интервал {args.schedule_interval/3600:.1f}ч).")
        if not args.no_upload:
            print("[i] Запускаю планировщик (Ctrl+C — сохранить состояние и выйти)...")
            scheduler_mod.run_schedule(
                interval=args.schedule_interval,
                state_file=scheduler_mod.DEFAULT_STATE_FILE,
                cookie_path=upload_mod.DEFAULT_COOKIE_FILE,
            )
    return 0


def _cmd_regen(args: argparse.Namespace) -> int:
    output_dir = args.output
    if output_dir == "cuts" and (args.series_title or args.episode != 1):
        from moneyprinter import serial as _serial
        output_dir = os.path.join("cuts", _serial._episode_dir(args.series_title, args.episode))
    n = serial_mod.regenerate_captions(
        input_path=args.input,
        output_dir=output_dir,
        series_title=args.series_title,
        episode=args.episode,
        base_hashtags=args.base_hashtag or None,
        global_hashtags=args.global_hashtag or None,
        whisper_model=args.whisper_model,
        device=args.device,
        language=args.language,
        llm_model=args.llm,
        llm_url=args.llm_url,
        auto_install=True,
        jobs=args.jobs,
    )
    print(f"Готово: перегенерировано описаний для {n} клипов.")
    return 0


def _build_enhance_cfg(args: argparse.Namespace) -> enhance_mod.EnhanceConfig:
    return enhance_mod.EnhanceConfig(
        target_width=getattr(args, "target_width", 1920),
        target_height=getattr(args, "target_height", 1080),
        crf=getattr(args, "crf", 18),
        preset=getattr(args, "preset", "slow"),
        denoise_strength=getattr(args, "denoise_strength", 3),
        sharpen_strength=getattr(args, "sharpen_strength", 1.5),
        sharp_mode=getattr(args, "sharp_mode", "cas+unsharp"),
        preserve_aspect=not getattr(args, "no_preserve_aspect", False),
        use_ai=getattr(args, "ai", False),
        ai_model=getattr(args, "ai_model", "realesrgan-x4plus"),
        jobs=getattr(args, "jobs", 0),
    )


def _cmd_enhance(args: argparse.Namespace) -> int:
    cfg = _build_enhance_cfg(args)
    input_path = args.input
    output_dir = args.output

    if os.path.isdir(input_path):
        enhance_mod.enhance_directory(input_path, output_dir, cfg)
    else:
        out_path = os.path.join(output_dir, Path(input_path).stem + "_enhanced.mp4")
        os.makedirs(output_dir, exist_ok=True)
        info = probe(input_path)
        print(f"[i] Улучшение: {input_path} ({info.width}x{info.height}, {info.duration:.1f}с)")
        enhance_mod.enhance_video(input_path, out_path, cfg, total_duration=info.duration)
        print(f"[✓] Готово: {out_path}")
    return 0


def _quiet_warnings() -> None:
    """Глушим сторонние warning-логи (HF, urllib3 и т.п.) в консоли."""
    import logging as _l
    import warnings as _w
    _l.getLogger("huggingface_hub").setLevel(_l.ERROR)
    _l.getLogger("urllib3").setLevel(_l.ERROR)
    _w.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
    _w.filterwarnings("ignore", message=".*symlinks by default.*")
    _w.filterwarnings("ignore", message=".*cache-system uses symlinks.*")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def main(argv=None) -> int:
    _quiet_warnings()
    args = _build_parser().parse_args(argv)
    try:
        # Команды, работающие с TikTok, сами обеспечат наличие Playwright
        if args.command in ("login", "publish", "run-schedule", "publish-next"):
            upload_mod.ensure_playwright(auto_install=True)

        if args.command == "probe":
            return _cmd_probe(args)
        if args.command == "login":
            return _cmd_login(args)
        if args.command == "publish":
            return _cmd_publish(args)
        if args.command == "run-schedule":
            return _cmd_run_schedule(args)
        if args.command == "publish-next":
            return _cmd_publish_next(args)
        if args.command == "install-scheduler":
            return _cmd_install_scheduler(args)
        if args.command == "uninstall-scheduler":
            return _cmd_uninstall_scheduler(args)
        if args.command == "serial":
            return _cmd_serial(args)
        if args.command == "regen-captions":
            return _cmd_regen(args)
        if args.command == "enhance":
            return _cmd_enhance(args)

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
            enhance=args.enhance,
            enhance_cfg=_build_enhance_cfg(args) if getattr(args, "enhance", False) else None,
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