"""Браузерная автоматизация загрузки в TikTok через Playwright.

ВНИМАНИЕ: TikTok не даёт публичного API для автопостинга. Этот модуль
имитирует действия пользователя на https://www.tiktok.com/upload. Это
нарушает ToS TikTok и может сломаться при смене верстки — используйте
на свой страх и риск, с умеренной частотой публикаций.

Поток работы:
  1. Зарегистрироваться один раз: `moneyprinter login` сохранит cookies
     в файл (после ручного входа по QR/логину в браузере).
  2. `moneyprinter publish <video> [--caption ...]` — выкладывает один
     ролик под сохранённой сессией.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

UPLOAD_URL = "https://www.tiktok.com/upload"
DEFAULT_COOKIE_FILE = os.path.expanduser("~/.moneyprinter/tiktok_cookies.json")


class UploadError(RuntimeError):
    pass


def _require_playwright() -> None:
    try:
        import playwright  # noqa: F401
    except ImportError as exc:
        raise UploadError(
            "Playwright не установлен. Выполните: pip install 'moneyprinter[upload]' "
            "затем: playwright install chromium"
        ) from exc


def save_cookies(cookies: list, path: str = DEFAULT_COOKIE_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cookies, ensure_ascii=False), encoding="utf-8")


def load_cookies(path: str = DEFAULT_COOKIE_FILE) -> list:
    if not os.path.exists(path):
        raise UploadError(f"Файл cookies не найден: {path}. Сначала выполните `moneyprinter login`.")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _launch(headed: bool):
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=not headed)
    context = browser.new_context(
        viewport={"width": 1280, "height": 900},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
    )
    return pw, browser, context


def interactive_login(path: str = DEFAULT_COOKIE_FILE, headed: bool = True, timeout: int = 180) -> None:
    """Открывает TikTok, ждёт ручного входа (QR/логин), сохраняет cookies."""
    _require_playwright()
    pw, browser, context = _launch(headed)
    try:
        page = context.new_page()
        print(f"[i] Открываю {UPLOAD_URL}. Войдите в аккаунт вручную (QR/логин).")
        page.goto(UPLOAD_URL, wait_until="domcontentloaded")
        # Ждём, пока пользователь не окажется залогиненным (пропадёт кнопка входа)
        try:
            page.wait_for_selector("text=Log in", timeout=timeout * 1000)
            print("[i] Жду, пока вы войдёте... (оставайтесь на странице)")
            page.wait_for_selector("text=Log in", state="detached", timeout=timeout * 1000)
        except Exception:
            pass
        cookies = context.cookies()
        save_cookies(cookies, path)
        print(f"[✓] Cookies сохранены: {path}")
    finally:
        browser.close()
        pw.stop()


def upload_video(
    video_path: str,
    caption: str = "",
    cookie_path: str = DEFAULT_COOKIE_FILE,
    headed: bool = False,
    timeout: int = 300,
) -> str:
    """Загружает один ролик на TikTok и публикует его.

    Возвращает caption, который был отправлен.
    """
    _require_playwright()
    cookies = load_cookies(cookie_path)
    pw, browser, context = _launch(headed)
    try:
        context.add_cookies(cookies)
        page = context.new_page()
        page.goto(UPLOAD_URL, wait_until="domcontentloaded")

        # 1) выбрать файл через input[type=file]
        file_input = page.wait_for_selector("input[type=file]", timeout=timeout * 1000)
        if file_input is None:
            raise UploadError("Не найден input для загрузки файла.")
        file_input.set_input_files(os.path.abspath(video_path))

        # 2) дождаться окончания обработки/загрузки (пропадает индикатор прогресса)
        print("[i] Загружаю видео на сервер TikTok...")
        page.wait_for_function(
            """() => {
                const el = document.querySelector('[data-e2e=upload-progress]');
                return !el || el.offsetParent === null;
            }""",
            timeout=timeout * 1000,
        )
        # небольшая пауза, чтобы появилось поле caption
        time.sleep(3)

        # 3) заполнить подпись
        caption_box = page.query_selector("div[contenteditable=true]")
        if caption_box is None:
            # запасной селектор
            caption_box = page.query_selector("#root textarea")
        if caption_box is not None and caption:
            caption_box.click()
            caption_box.type(caption, delay=20)

        # 4) нажать «Post» / «Опубликовать»
        posted = False
        for sel in ["button:has-text('Post')", "button:has-text('Опубликовать')",
                    "div[role=button]:has-text('Post')"]:
            btn = page.query_selector(sel)
            if btn is not None:
                btn.click()
                posted = True
                break
        if not posted:
            raise UploadError("Не найдена кнопка публикации. Возможно, изменилась вёрстка TikTok.")

        # 5) дождаться подтверждения
        try:
            page.wait_for_selector("text=Your video is being processed", timeout=60000)
        except Exception:
            pass
        print(f"[✓] Опубликовано: {os.path.basename(video_path)}")
        return caption
    finally:
        browser.close()
        pw.stop()
