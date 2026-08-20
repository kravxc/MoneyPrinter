"""CLI интерфейс MoneyPrinter."""

from __future__ import annotations

import argparse
import sys

from .pipeline import Config, process
from .media import probe


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
    p_run.add_argument("--min-duration", type=float, default=10.0, help="Мин. длина клипа, сек")
    p_run.add_argument("--max-duration", type=float, default=180.0, help="Макс. длина клипа, сек (Shorts допускает до 180)")
    p_run.add_argument("--story-gap", type=float, default=2.0, help="Макс. пауза между репликами одной мысли, сек")
    p_run.add_argument("--min-score", type=float, default=0.0, help="Отсечка по суммарному скору")
    p_run.add_argument("--horizontal", action="store_true", help="Не конвертировать в 9:16 (оставить пропорции)")
    p_run.add_argument("--crop", action="store_true", help="Вертикаль кадрированием вместо размытого фона")
    p_run.add_argument("--whisper-model", default="base", help="Модель Whisper: tiny/base/small/medium/large")
    p_run.add_argument("--device", default="auto", help="auto/cpu/cuda (auto = cpu; cuda ускорит на NVIDIA GPU)")
    p_run.add_argument("--jobs", type=int, default=0, help="Сколько клипов нарезать параллельно (0 = все ядра)")
    p_run.add_argument("--no-auto-install", action="store_true", help="Не доустанавливать AI-зависимости автоматически")
    p_run.add_argument("--keep-ads", action="store_true", help="Не вырезать визуальные баннеры казино/беттинга из клипов")
    p_run.add_argument("--language", default=None, help="Язык для транскрипции (напр. ru, en)")
    p_run.add_argument("--llm", default=None, metavar="MODEL", help="Локальная LLM для ранжирования (напр. llama3.2)")
    p_run.add_argument("--llm-url", default=None, help="Ollama base url (default: http://localhost:11434)")
    p_run.add_argument("--scene-threshold", type=float, default=27.0, help="Чувствительность детекции сцен")

    p_probe = sub.add_parser("probe", help="Показать метаданные видео")
    p_probe.add_argument("input", help="Путь к видеофайлу")

    return parser


def _cmd_probe(args: argparse.Namespace) -> int:
    info = probe(args.input)
    print(f"path:     {info.path}")
    print(f"duration: {info.duration:.2f}s")
    print(f"size:     {info.width}x{info.height}")
    print(f"fps:      {info.fps:.2f}")
    print(f"audio:    {'yes' if info.has_audio else 'no'}")
    return 0


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "probe":
            return _cmd_probe(args)
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
            remove_ads=not args.keep_ads,
        )
        result = process(cfg)
        print(f"\nГотово: {len(result.clips)} клипов → {args.output}")
        return 0
    except Exception as exc:  # noqa: BLE001 — CLI должен показать дружелюбную ошибку
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())