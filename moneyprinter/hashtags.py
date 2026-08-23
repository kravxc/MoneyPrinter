"""Генерация хештегов и caption для TikTok/Shorts.

Два источника:
  * базовый набор — всегда добавляется для охвата (#shorts, #фильм и т.п.);
  * «умные» теги — либо через локальную LLM (как в score.rank_with_llm),
    либо эвристикой по ключевым словам текста клипа (фолбэк без LLM).
"""

from __future__ import annotations

import re
from typing import List, Optional

# Базовый набор — не зависит от содержимого, даёт охват по платформе/жанру
BASE_HASHTAGS = [
    "shorts",
    "тикток",
    "фильм",
    "сериал",
    "момент",
]

# Языковые маркеры → теги (простая эвристика, если нет LLM)
_TOPIC_TAGS_RU = [
    ("смех", "смешно"),
    ("шутк", "шутка"),
    ("любов", "любовь"),
    ("драм", "драма"),
    ("ужас", "ужасы"),
    ("боев", "боевик"),
    ("реакц", "реакция"),
    ("топ", "топ"),
    ("офиг", "офигенно"),
    ("жесть", "жесть"),
    ("игра", "игры"),
    ("спорт", "спорт"),
    ("музык", "музыка"),
    ("дет", "дети"),
    ("семь", "семья"),
]
_TOPIC_TAGS_EN = [
    ("funny", "funny"),
    ("lol", "lol"),
    ("love", "love"),
    ("drama", "drama"),
    ("scary", "scary"),
    ("fight", "fight"),
    ("best", "best"),
    ("game", "gaming"),
    ("music", "music"),
    ("sport", "sport"),
]


def _clean_tag(tag: str) -> Optional[str]:
    """Нормализует тег: lowercase, без #, пробелов, спецсимволов (кроме _)."""
    tag = tag.strip().lower()
    tag = tag.lstrip("#")
    tag = re.sub(r"[^\wа-яё]+", "", tag, flags=re.UNICODE)
    if not tag or len(tag) > 30:
        return None
    return tag


def _keyword_fallback(text: str, limit: int) -> List[str]:
    """Эвристика без LLM: теги по ключевым словам + частым словам текста."""
    low = (text or "").lower()
    tags: List[str] = []
    for needle, tag in _TOPIC_TAGS_RU + _TOPIC_TAGS_EN:
        if needle in low and tag not in tags:
            tags.append(tag)
    # частые осмысленные слова (длина >= 4)
    words = re.findall(r"[a-zа-яё]{4,}", low, flags=re.UNICODE)
    stop = {
        "этот", "такой", "котор", "потом", "сейчас", "здесь", "там", "говор",
        "сказал", "знаешь", "просто", "вообще", "понима", "чтобы", "потому",
        "this", "that", "with", "your", "have", "what", "they", "when", "from",
    }
    freq: dict = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:limit]:
        if w not in tags:
            tags.append(w)
    return tags[:limit]


def _llm_tags(text: str, model: str, base_url: Optional[str], limit: int) -> Optional[List[str]]:
    """Просит локальную LLM выдать до `limit` релевантных тегов."""
    try:
        import ollama
    except ImportError:
        return None

    client = ollama.Client(host=base_url) if base_url else ollama
    prompt = (
        "Ты — SMM-щик TikTok. По тексту видео придумай до "
        f"{limit} подходящих хештегов (без #, по одному на строку), "
        "только слова/фразы на языке текста. Никаких пояснений.\n"
        f"Текст: {text!r}\n"
        "Хештеги:"
    )
    try:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.4, "num_predict": 80},
        )
        raw = resp["message"]["content"]
        tags = []
        for line in raw.splitlines():
            line = line.strip(" -\t*.")
            t = _clean_tag(line)
            if t and t not in tags:
                tags.append(t)
        return tags[:limit]
    except Exception:
        return None


def generate_hashtags(
    text: str,
    base: Optional[List[str]] = None,
    llm_model: Optional[str] = None,
    llm_url: Optional[str] = None,
    max_smart: int = 5,
) -> List[str]:
    """Возвращает итоговый список хештегов (без #) для клипа.

    Сначала пытается взять «умные» теги из LLM, при неудаче — эвристика
    по ключевым словам. Поверх всегда добавляется базовый набор.
    """
    smart: List[str] = []
    if llm_model:
        smart = _llm_tags(text, llm_model, llm_url, max_smart) or []
    if not smart:
        smart = _keyword_fallback(text, max_smart)

    out: List[str] = []
    for t in (base or BASE_HASHTAGS):
        ct = _clean_tag(t)
        if ct and ct not in out:
            out.append(ct)
    for t in smart:
        if t not in out:
            out.append(t)
    return out


def build_caption(text: str, hashtags: List[str], max_chars: int = 2000) -> str:
    """Собирает подпись: короткий текст-цитата + хештеги строкой."""
    caption_bits = []
    if text:
        snippet = text.strip().replace("\n", " ")
        snippet = (snippet[:120] + "…") if len(snippet) > 120 else snippet
        caption_bits.append(snippet)
    tags = " ".join(f"#{t}" for t in hashtags)
    if tags:
        caption_bits.append(tags)
    caption = "\n\n".join(caption_bits)
    return caption[:max_chars]


def _extract_hook_fallback(text: str, limit: int = 120) -> str:
    """Без LLM: берём самую «интересную» фразу — длинную и осмысленную."""
    import re

    sentences = re.split(r"[.!?…]+", (text or "").replace("\n", " "))
    sentences = [s.strip(" -\t") for s in sentences if len(s.strip()) >= 4]
    if not sentences:
        return (text or "").strip()[:limit]
    for needle in ("смех", "ха", "lol", "ого", "вау", "шок", "ужас", "кто", "почему", "зачем"):
        for s in sentences:
            if needle in s.lower():
                return s[:limit]
    return max(sentences, key=len)[:limit]


def generate_hook(
    text: str,
    llm_model: Optional[str] = None,
    llm_url: Optional[str] = None,
    limit: int = 120,
) -> str:
    """Короткий «крючок»-описание по содержанию (вопрос/интрига, 1 предложение).

    Сначала пробуем локальную LLM (как в rank_with_llm), при неудаче —
    эвристика по тексту. Пустой текст → пустая строка.
    """
    text = (text or "").strip()
    if not text:
        return ""
    if llm_model:
        try:
            import ollama
        except ImportError:
            return _extract_hook_fallback(text, limit)
        client = ollama.Client(host=llm_url) if llm_url else ollama
        prompt = (
            "Ты — редактор TikTok. По тексту видео придумай ОДИН цепляющий "
            "крючок-описание: короткий вопрос или интрига, максимум 120 символов, "
            "на языке текста, без хештегов и кавычек. Только сама фраза.\n"
            f"Текст: {text!r}\nКрючок:"
        )
        try:
            resp = client.chat(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.5, "num_predict": 80},
            )
            hook = resp["message"]["content"].strip().strip('"').strip("'")
            hook = hook.splitlines()[0] if hook else ""
            if hook:
                return hook[:limit]
        except Exception:
            pass
    return _extract_hook_fallback(text, limit)
