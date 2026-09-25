"""Скачивание видео через yt-dlp (обёртка).

Используется для загрузки собственных роликов/шортсов в исходном качестве.
В отличие от web-сервисов-«спасателей» (get-save и т.п.), которые отдают
пережатые стримы в маленьком разрешении (и из-за этого TikTok режет контент
как «низкокачественный»), yt-dlp берёт исходный поток с YouTube напрямую.

Если yt-dlp не установлен — он ставится сам через pip (работает и на
Windows, и на macOS/Linux).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

DOWNLOAD_DIR_DEFAULT = "input"


class FetchError(RuntimeError):
    pass


def _ensure_ytdlp(auto_install: bool) -> bool:
    """Проверяет наличие yt-dlp; при auto_install сам ставит его через pip."""
    try:
        import yt_dlp  # noqa: F401

        return True
    except ImportError:
        pass
    if not auto_install:
        return False
    print("[i] yt-dlp не найден — устанавливаю (первый раз может занять минуту)...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "yt-dlp"]
        )
        import yt_dlp  # noqa: F401

        return True
    except Exception as exc:  # noqa: BLE001
        raise FetchError(
            "Не удалось установить yt-dlp. Выполните вручную: pip install yt-dlp"
        ) from exc


def _progress_hook(data: dict) -> None:
    """Показывает прогресс скачивания в консоли (одна строка, обновляется)."""
    status = data.get("status")
    if status == "downloading":
        pct = (data.get("_percent_str") or "").strip()
        speed = (data.get("_speed_str") or "").strip()
        eta = (data.get("_eta_str") or "").strip()
        if pct:
            print(f"\r  [fetch] {pct}  {speed}  {eta}", end="", flush=True)
    elif status == "finished":
        print("\r  [✓] Скачано.                           ")


def _build_opts(
    output_dir: str, max_height: int
) -> dict:
    """Опции для yt_dlp: исходное качество, h264 (не требует перекодировки)."""
    return {
        # лучший видео-поток до max_height + аудио; иначе лучший доступный
        "format": f"bv*[height<={max_height}]+ba/bv*[height<={max_height}]/bv*+ba/b",
        # предпочитать по разрешению, кодек h264 (совместим с ffmpeg-merge)
        "format_sort": ["res", "vcodec:h264"],
        "merge_output_format": "mp4",
        "outtmpl": str(Path(output_dir) / "%(title)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "no_color": True,  # на Windows цветные коды в логах не нужны
        "progress_hooks": [_progress_hook],
        "retries": 3,
    }


def fetch_video_urls(
    urls: List[str],
    output_dir: str = DOWNLOAD_DIR_DEFAULT,
    max_height: int = 1080,
    auto_install: bool = True,
) -> List[str]:
    """Скачивает список URL в output_dir, возвращает пути скачанных файлов."""
    if not _ensure_ytdlp(auto_install):
        raise FetchError(
            "yt-dlp не установлен. Выполните: pip install yt-dlp "
            "или запустите без --no-auto-install."
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    import yt_dlp

    opts = _build_opts(str(out), max_height)
    downloaded: List[str] = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        for url in urls:
            print(f"\n[→] Скачиваю: {url}")
            try:
                info = ydl.extract_info(url, download=True)
            except yt_dlp.utils.DownloadError as exc:
                raise FetchError(f"Ошибка загрузки {url}: {exc}") from exc

            paths = []
            for req in info.get("requested_downloads") or []:
                fp = req.get("filepath")
                if fp:
                    paths.append(fp)
            if not paths:
                fp = info.get("_filename") or info.get("filepath")
                if fp:
                    paths.append(fp)
            for fp in paths:
                downloaded.append(fp)
                print(f"  [✓] Сохранено: {fp}")
    return downloaded