"""Планировщик публикаций в TikTok.

Логика (как вы хотели):
  * первый клип из очереди публикуется сразу;
  * остальные ставятся в расписание с интервалом (по умолчанию 2 часа,
    настраивается --schedule-interval, сек);
  * состояние сохраняется в JSON-файл, чтобы переживать перезапуск
    процесса: уже опубликованные не повторяются, отложенные публикуются
    по наступлении своего времени.

Запуск: `moneyprinter run-schedule` держит очередь в памяти и публикует
по таймеру. Можно запускать в tmux/screen — при падении и рестарте
неопубликованное продолжит выкладываться.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from .upload import upload_video

DEFAULT_STATE_FILE = os.path.expanduser("~/.moneyprinter/schedule.json")
DEFAULT_INTERVAL = 7200  # 2 часа


@dataclass
class QueueItem:
    video_path: str
    caption: str = ""
    scheduled_at: float = 0.0  # unix-time; 0 = ещё не назначено
    published: bool = False
    published_at: float = 0.0
    error: str = ""


@dataclass
class ScheduleState:
    items: List[QueueItem] = field(default_factory=list)
    interval: float = DEFAULT_INTERVAL
    state_file: str = DEFAULT_STATE_FILE

    def save(self) -> None:
        Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.state_file).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str, interval: float) -> "ScheduleState":
        if os.path.exists(path):
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            items = [QueueItem(**i) for i in data.get("items", [])]
            return cls(items=items, interval=data.get("interval", interval), state_file=path)
        return cls(items=[], interval=interval, state_file=path)


def plan_queue(
    videos: List[str],
    captions: Optional[List[str]] = None,
    interval: float = DEFAULT_INTERVAL,
    state_file: str = DEFAULT_STATE_FILE,
) -> ScheduleState:
    """Формирует очередь: 1-й неопубликованный — сразу (now), остальные — через интервал каждый.

    Если state_file уже существует, дополняет его новыми видео, сохраняя
    уже запланированные/опубликованные (по абсолютному пути видео).
    """
    captions = captions or [""] * len(videos)
    state = ScheduleState.load(state_file, interval)
    state.interval = interval
    existing = {i.video_path for i in state.items}

    now = time.time()
    # база для новых элементов: сразу после последнего запланированного
    last_sched = max(
        [i.scheduled_at for i in state.items if i.scheduled_at], default=now
    )
    cursor = last_sched
    # первый НЕопубликованный элемент (с учётом уже существующих) идёт сразу
    has_unpublished = any(not i.published for i in state.items)
    for video, cap in zip(videos, captions):
        if video in existing:
            continue
        if not has_unpublished:
            sched = now
            has_unpublished = True
        else:
            cursor += interval
            sched = max(cursor, now)
        state.items.append(QueueItem(video_path=video, caption=cap, scheduled_at=sched))
    state.save()
    return state


def _next_due(state: ScheduleState) -> Optional[QueueItem]:
    now = time.time()
    due = [
        i for i in state.items
        if not i.published and i.scheduled_at <= now
    ]
    if not due:
        return None
    return min(due, key=lambda i: i.scheduled_at)


def publish_next(
    interval: float = DEFAULT_INTERVAL,
    state_file: str = DEFAULT_STATE_FILE,
    cookie_path: str = "",
    dry_run: bool = False,
) -> bool:
    """One-shot: публикует ровно один «наступивший» клип и выходит.

    Предназначен для запуска из системного планировщика (cron/launchd/
    schtasks), чтобы программа НЕ висела в памяти постоянно. Возвращает
    True, если что-то опубликовано (или dry-run), иначе False.

    При ошибке публикации клип остаётся «due» (scheduled_at ставится в
    прошлое на интервал), чтобы системный планировщик повторил попытку
    при следующем просыпании, а не ждал целый интервал.
    """
    state = ScheduleState.load(state_file, interval)
    state.interval = interval
    item = _next_due(state)
    if item is None:
        pending = [i for i in state.items if not i.published]
        if not pending:
            print("[i] Очередь пуста — публиковать нечего.")
        else:
            soon = min(pending, key=lambda i: i.scheduled_at)
            wait = max(0.0, soon.scheduled_at - time.time())
            print(f"[i] Рано. Следующий клип будет готов через {wait/60:.1f} мин.")
        return False

    print(f"[→] Публикую: {os.path.basename(item.video_path)}")
    if dry_run:
        print(f"    (dry-run) caption: {item.caption[:80]}")
        item.published = True
        item.published_at = time.time()
        state.save()
        return True
    try:
        upload_video(item.video_path, item.caption, cookie_path=cookie_path or "")
        item.published = True
        item.published_at = time.time()
        state.save()
        return True
    except Exception as exc:  # noqa: BLE001
        item.error = str(exc)
        print(f"[!] Ошибка публикации {os.path.basename(item.video_path)}: {exc}")
        # оставляем due в прошлом, чтобы системный планировщик повторил скоро
        item.scheduled_at = time.time() - interval
        state.save()
        return False


def install_scheduler(
    interval: float = DEFAULT_INTERVAL,
    state_file: str = DEFAULT_STATE_FILE,
    cookie_path: str = "",
    dry_run: bool = False,
) -> str:
    """Создаёт задание в системном планировщике (cron/launchd/schtasks).

    Программа сама НЕ работает постоянно — системный планировщик будит
    `moneyprinter publish-next` каждые `interval` секунд. Поддерживаются:
      * macOS / Linux  → launchd (macOS, user agent) или crontab (Linux);
      * Windows        → schtasks.

    Возвращает человекочитаемое описание того, что сделано.
    """
    import shutil
    import subprocess

    exe = shutil.which("moneyprinter") or f"{sys.executable} -m moneyprinter"
    cmd = f'{exe} publish-next --state "{state_file}" --cookie "{cookie_path}" --schedule-interval {interval:g}'

    minutes = max(1, int(round(interval / 60)))

    if sys.platform == "win32":
        task = "MoneyPrinterTikTok"
        schtasks = (
            f'schtasks /Create /TN "{task}" /TR "{cmd}" '
            f"/SC MINUTE /MO {minutes} /F"
        )
        if dry_run:
            return f"(dry-run) Windows schtasks:\n  {schtasks}"
        subprocess.run(schtasks, shell=True, check=False)
        return f"Создана задача Windows «{task}»: запуск каждые {minutes} мин."

    if sys.platform == "darwin":
        label = "com.moneyprinter.tiktok"
        plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{label}.plist")
        prog = shutil.which("moneyprinter") or sys.executable
        prog_args = [prog]
        if shutil.which("moneyprinter") is None:
            prog_args = [sys.executable, "-m", "moneyprinter"]
        plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        {chr(10).join(f"        <string>{a}</string>" for a in prog_args)}
        <string>publish-next</string>
        <string>--state</string><string>{state_file}</string>
        <string>--cookie</string><string>{cookie_path}</string>
        <string>--schedule-interval</string><string>{interval:g}</string>
    </array>
    <key>StartInterval</key><integer>{int(interval)}</integer>
    <key>RunAtLoad</key><true/>
    <key>StandardOutPath</key><string>{os.path.expanduser('~/.moneyprinter/launchd.log')}</string>
    <key>StandardErrorPath</key><string>{os.path.expanduser('~/.moneyprinter/launchd.err')}</string>
</dict>
</plist>
"""
        if dry_run:
            return f"(dry-run) macOS launchd plist → {plist_path}:\n{plist}"
        Path(plist_path).parent.mkdir(parents=True, exist_ok=True)
        Path(plist_path).write_text(plist, encoding="utf-8")
        subprocess.run(["launchctl", "load", plist_path], check=False)
        return f"launchd agent «{label}» установлен: запуск каждые {minutes} мин."

    # Linux — crontab
    cron_line = f"*/{minutes} * * * * {cmd} >> {os.path.expanduser('~/.moneyprinter/cron.log')} 2>&1"
    if dry_run:
        return f"(dry-run) Linux crontab:\n  {cron_line}"
    try:
        existing = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False).stdout or ""
    except Exception:
        existing = ""
    marker = "# moneyprinter-tiktok"
    kept = [l for l in existing.splitlines() if marker not in l]
    kept.append(f"{marker}\n{cron_line}")
    subprocess.run(["crontab", "-"], input="\n".join(kept) + "\n", text=True, check=False)
    return f"Добавлена строка в crontab: запуск каждые {minutes} мин."


def run_schedule(
    interval: float = DEFAULT_INTERVAL,
    state_file: str = DEFAULT_STATE_FILE,
    cookie_path: str = "",
    dry_run: bool = False,
    poll: float = 30.0,
) -> None:
    """Держит очередь и публикует по расписанию.

    Первый due-элемент (обычно с scheduled_at=now) уходит сразу,
    следующие — по прошествии интервала между ними.
    """
    state = ScheduleState.load(state_file, interval)
    state.interval = interval
    if not state.items:
        print("[i] Очередь пуста. Добавьте видео через `moneyprinter publish --queue`.")
        return
    print(f"[i] Очередь: {len(state.items)} видео, интервал {interval/3600:.1f}ч. Ctrl+C для выхода.")

    try:
        while True:
            pending = [i for i in state.items if not i.published]
            if not pending:
                print("[✓] Все видео опубликованы.")
                break
            item = _next_due(state)
            if item is None:
                soonest = min(pending, key=lambda i: i.scheduled_at)
                wait = max(0.0, soonest.scheduled_at - time.time())
                print(f"[i] Следующая публикация через {wait/60:.1f} мин: {os.path.basename(soonest.video_path)}")
                time.sleep(min(poll, wait + 1))
                continue

            print(f"[→] Публикую: {os.path.basename(item.video_path)}")
            if dry_run:
                print(f"    (dry-run) caption: {item.caption[:80]}")
                item.published = True
                item.published_at = time.time()
            else:
                try:
                    upload_video(item.video_path, item.caption, cookie_path=cookie_path or "")
                    item.published = True
                    item.published_at = time.time()
                except Exception as exc:  # noqa: BLE001 — продолжаем очередь при ошибке
                    item.error = str(exc)
                    print(f"[!] Ошибка публикации {os.path.basename(item.video_path)}: {exc}")
                    # сдвигаем на интервал, чтобы не спамить при ошибке
                    item.scheduled_at = time.time() + interval
            state.save()
            time.sleep(poll)
    except KeyboardInterrupt:
        state.save()
        print("\n[i] Сохранено состояние очереди. Перезапуск `run-schedule` продолжит.")
