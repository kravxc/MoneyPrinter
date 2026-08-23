"""Тесты генерации хештегов и планировщика очереди."""

from __future__ import annotations

import time

from moneyprinter import hashtags, scheduler


def test_clean_tag_strips_junk():
    assert hashtags._clean_tag("#Смех!") == "смех"
    assert hashtags._clean_tag("  Funny ") == "funny"
    assert hashtags._clean_tag("") is None


def test_keyword_fallback_finds_topic():
    tags = hashtags._keyword_fallback("это было очень смешно и смех", limit=5)
    assert "смешно" in tags or "смех" in tags


def test_generate_hashtags_includes_base():
    tags = hashtags.generate_hashtags("просто текст без маркеров", llm_model=None)
    assert "shorts" in tags
    assert "тикток" in tags


def test_build_caption_formats_tags():
    cap = hashtags.build_caption("привет мир", ["шутка", "смех"])
    assert "#шутка" in cap and "#смех" in cap
    assert "привет мир" in cap


def test_generate_hook_fallback_picks_interesting_sentence():
    text = "Обычный день. Кто отравил пробирку? Просто шли по коридору."
    hook = hashtags.generate_hook(text, llm_model=None)
    assert "пробирку" in hook


def test_generate_hook_empty_text():
    assert hashtags.generate_hook("", llm_model=None) == ""


def test_plan_queue_first_immediate_rest_spaced(tmp_path):
    state_file = str(tmp_path / "sched.json")
    interval = 3600.0
    vids = [f"/tmp/v{i}.mp4" for i in range(3)]
    caps = ["c"] * 3
    state = scheduler.plan_queue(vids, caps, interval=interval, state_file=state_file)
    assert len(state.items) == 3
    # первый — сразу (в пределах секунды от now)
    assert abs(state.items[0].scheduled_at - time.time()) < 2
    # остальные — с шагом interval
    assert state.items[1].scheduled_at - state.items[0].scheduled_at >= interval - 2
    assert state.items[2].scheduled_at - state.items[1].scheduled_at >= interval - 2


def test_plan_queue_dedup_existing(tmp_path):
    state_file = str(tmp_path / "sched.json")
    vids = ["/tmp/a.mp4", "/tmp/b.mp4"]
    scheduler.plan_queue(vids, ["c", "c"], interval=100, state_file=state_file)
    # повторный вызов с тем же + новым — дубликатов нет
    state = scheduler.plan_queue(vids + ["/tmp/c.mp4"], ["c"] * 3, interval=100, state_file=state_file)
    paths = [i.video_path for i in state.items]
    assert paths.count("/tmp/a.mp4") == 1
    assert paths.count("/tmp/c.mp4") == 1
    assert len(state.items) == 3
