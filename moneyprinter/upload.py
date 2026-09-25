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
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

UPLOAD_URL = "https://www.tiktok.com/upload"
DEFAULT_COOKIE_FILE = os.path.expanduser("~/.moneyprinter/tiktok_cookies.json")

DEFAULT_PROFILE_DIR = os.path.expanduser("~/.moneyprinter/tiktok_profile")


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

STEALTH_JS = (
    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
    "window.navigator.chrome = {runtime: {}};"
    "Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3]});"
    "Object.defineProperty(navigator, 'languages', {get: () => ['ru-RU','ru','en']});"

    "const __proto = Object.getPrototypeOf(navigator);"
    "delete __proto.webdriver;"
    "for (const k of Object.getOwnPropertyNames(__proto)) {"
    "  if (k.toLowerCase().includes('cdc') || k.toLowerCase().includes('playwright')) {"
    "    try { delete __proto[k]; } catch (e) {}"
    "  }"
    "}"
)


class UploadError(RuntimeError):
    pass


def ensure_playwright(auto_install: bool = True) -> bool:
    """Проверяет Playwright и браузер; при auto_install сам ставит недостающее.

    Возвращает True, если всё готово к запуску.
    """
    try:
        import playwright  # noqa: F401
    except ImportError:
        if not auto_install:
            return False
        print("[i] Playwright не найден — устанавливаю...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", "playwright"]
            )
        except Exception as exc:
            raise UploadError(
                "Не удалось установить Playwright. Выполните вручную: "
                "pip install playwright && python -m playwright install chromium"
            ) from exc


    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:

            _ = pw.chromium.executable_path
        return True
    except Exception:
        if not auto_install:
            return False
        print("[i] Браузер Chromium не найден — скачиваю (может занять пару минут)...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "playwright", "install", "chromium"]
            )
            return True
        except Exception as exc:
            raise UploadError(
                "Не удалось скачать Chromium. Выполните вручную: "
                "python -m playwright install chromium"
            ) from exc


def _require_playwright() -> None:
    if not ensure_playwright(auto_install=True):
        raise UploadError(
            "Playwright не установлен. Выполните: pip install 'moneyprinter[upload]' "
            "затем: python -m playwright install chromium"
        )


def save_cookies(cookies: list, path: str = DEFAULT_COOKIE_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cookies, ensure_ascii=False), encoding="utf-8")


def load_cookies(path: str = DEFAULT_COOKIE_FILE) -> list:
    if not os.path.exists(path):
        raise UploadError(f"Файл cookies не найден: {path}. Сначала выполните `moneyprinter login`.")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _launch(headed: bool, profile_dir: Optional[str] = None, prefer_chrome: bool = True):
    """Запускает браузер. Если задан profile_dir — использует persistent-профиль.

    Persistent-профиль критичен для входа через Google: в нём остаётся
    «живой» браузер, и Google не считает его автоматизированным/небезопасным.

    По умолчанию пытаемся использовать реальный Google Chrome (channel="chrome"),
    т.к. Google/TikTok доверяют ему гораздо больше, чем Playwright-Chromium, и
    реже выдают «слишком много попыток». Если Chrome не установлен — фолбэк на
    бандл Chromium с максимальным сокрытием автоматизации.
    """
    _require_playwright()
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()

    ignore_default = ["--enable-automation"]
    launch_args = [
        "--disable-blink-features=AutomationControlled",
        "--no-sandbox",
        "--disable-infobars",
    ]

    def _build_context(context_kwargs: dict):
        context_kwargs.setdefault("viewport", {"width": 1280, "height": 900})
        context_kwargs.setdefault("user_agent", USER_AGENT)
        return context_kwargs

    if profile_dir:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        if prefer_chrome:
            try:
                context = pw.chromium.launch_persistent_context(
                    profile_dir,
                    channel="chrome",
                    headless=False,
                    ignore_default_args=ignore_default,
                    args=launch_args,
                    **_build_context({}),
                )
                print("[i] Запущен реальный Google Chrome (меньше риск блокировки).")
            except Exception as exc:
                print(f"[i] Chrome недоступен ({exc}); запускаю Playwright Chromium.")
                context = pw.chromium.launch_persistent_context(
                    profile_dir,
                    headless=False,
                    ignore_default_args=ignore_default,
                    args=launch_args,
                    **_build_context({}),
                )
        else:
            context = pw.chromium.launch_persistent_context(
                profile_dir,
                headless=False,
                ignore_default_args=ignore_default,
                args=launch_args,
                **_build_context({}),
            )
        browser = None
    else:
        browser = pw.chromium.launch(
            headless=not headed,
            ignore_default_args=ignore_default,
            args=launch_args,
        )
        context = browser.new_context(**_build_context({}))

    context.add_init_script(STEALTH_JS)
    return pw, browser, context


def interactive_login(
    path: str = DEFAULT_COOKIE_FILE,
    profile_dir: str = DEFAULT_PROFILE_DIR,
    timeout: int = 300,
    force_new: bool = False,
    prefer_chrome: bool = True,
) -> None:
    """Открывает TikTok в persistent-профиле, ждёт ручного входа (в т.ч. через Google).

    Профиль сохраняется на диск, так что вход «живёт» между запусками и
    Google не блокирует его как небезопасный. Cookies дополнительно
    дублируются в файл для быстрой выгрузки.
    """
    if force_new and os.path.isdir(profile_dir):
        shutil.rmtree(profile_dir, ignore_errors=True)
    pw, browser, context = _launch(headed=True, profile_dir=profile_dir, prefer_chrome=prefer_chrome)
    try:
        page = context.new_page()
        print(f"[i] Открываю {UPLOAD_URL}. Войдите в аккаунт вручную (QR / Google / телефон).")
        print("    Браузер «живой» (persistent-профиль), поэтому Google не блокирует вход.")
        page.goto(UPLOAD_URL, wait_until="domcontentloaded")
        try:
            page.wait_for_selector("text=Log in", timeout=timeout * 1000)
            print("[i] Жду, пока вы войдёте... (оставайтесь на странице)")
            page.wait_for_selector("text=Log in", state="detached", timeout=timeout * 1000)
        except Exception:
            pass

        time.sleep(3)
        cookies = context.cookies()
        save_cookies(cookies, path)
        print(f"[✓] Вход выполнен. Cookies сохранены: {path}")
        print(f"    Профиль (для повторного входа) лежит в: {profile_dir}")
    finally:
        context.close()
        if browser is not None:
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
    pw, browser, context = _launch(headed, prefer_chrome=True)
    try:
        context.add_cookies(cookies)
        page = context.new_page()
        page.goto(UPLOAD_URL, wait_until="domcontentloaded")


        file_input = page.wait_for_selector("input[type=file]", timeout=timeout * 1000)
        if file_input is None:
            raise UploadError("Не найден input для загрузки файла.")
        file_input.set_input_files(os.path.abspath(video_path))


        print("[i] Загружаю видео на сервер TikTok...")
        page.wait_for_function(
            """() => {
                const el = document.querySelector('[data-e2e=upload-progress]');
                return !el || el.offsetParent === null;
            }""",
            timeout=timeout * 1000,
        )

        time.sleep(3)


        caption_box = page.query_selector("div[contenteditable=true]")
        if caption_box is None:

            caption_box = page.query_selector("#root textarea")
        if caption_box is not None and caption:
            caption_box.click()
            caption_box.type(caption, delay=20)


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


        try:
            page.wait_for_selector("text=Your video is being processed", timeout=60000)
        except Exception:
            pass
        print(f"[✓] Опубликовано: {os.path.basename(video_path)}")
        return caption
    finally:
        browser.close()
        pw.stop()
