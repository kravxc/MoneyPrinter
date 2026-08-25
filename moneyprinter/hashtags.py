"""Генерация хештегов и caption для TikTok/Shorts.

Два источника:
  * базовый набор — всегда добавляется для охвата (#shorts, #фильм и т.п.);
  * «умные» теги — либо через локальную LLM (как в score.rank_with_llm),
    либо эвристикой по ключевым словам текста клипа (фолбэк без LLM).
"""

from __future__ import annotations

import re
from typing import List, Optional

# Базовый набор — самые популярные, реально используемые теги TikTok/Shorts
# (высокий охват, не «банные»). Всегда добавляются для видимости.
BASE_HASHTAGS = [
    # охватные «алгоритмические» теги (существуют и реально крутятся в TikTok)
    "fyp",          # самый популярный тег TikTok (For You Page)
    "fypシ",        # вариация fyp с японским символом (огромный охват)
    "foryou",       # вторая вариация fyp
    "foryoupage",
    "fypage",
    "viral",
    "viralvideo",
    "trending",
    "trend",
    "explore",
    "mustwatch",    # «обязательно к просмотру»
    "mustsee",
    "recommend",
    "reels",
    "shorts",
    # тематические (для нарезок фильмов/сериалов/видео)
    "movie",
    "movietok",
    "seriestok",
    "series",
    "recap",
    "movierecap",
    "entertainment",
    "bingewatch",
    "comedy",
    "funny",
    # русскоязычные
    "тикток",
    "тренд",
    "вирусное",
    "фильм",
    "сериал",
    "кино",
    "обзор",
]

# Реально популярные теги по темам (используются миллионами роликов).
# Ключевое слово из текста → список готовых популярных тегов.
_TOPIC_TAGS = [
    # жанры / настроение
    ("смех", ["смешно", "прикол", "комедия", "funny", "humor", "lol"]),
    ("шутк", ["шутка", "прикол", "комедия", "funny"]),
    ("любов", ["любовь", "романтика", "romance", "love"]),
    ("драм", ["драма", "melodrama", "drama"]),
    ("ужас", ["ужасы", "хоррор", "scary", "horror"]),
    ("страх", ["ужасы", "хоррор", "scary", "horror"]),
    ("боев", ["боевик", "экшн", "action", "fight"]),
    ("детектив", ["детектив", "mystery", "crime"]),
    ("убийств", ["детектив", "mystery", "crime"]),
    ("тайна", ["тайна", "mystery"]),
    ("загадк", ["загадка", "mystery"]),
    ("полиц", ["полиция", "crime", "detective"]),
    ("суд", ["суд", "law", "crime"]),
    ("преступл", ["криминал", "crime", "detective"]),
    ("триллер", ["триллер", "thriller", "suspense"]),
    ("хими", ["наука", "chemistry", "science", "школа"]),
    ("лаборатор", ["наука", "chemistry", "experiment"]),
    ("наука", ["наука", "science", "knowledge"]),
    ("школ", ["школа", "учеба", "school", "study"]),
    ("учеб", ["учеба", "school", "study", "знания"]),
    ("университет", ["университет", "study", "school"]),
    ("студент", ["студент", "study", "university"]),
    ("игра", ["игры", "gaming", "gameplay", "gamer"]),
    ("спорт", ["спорт", "sport", "fitness", "workout"]),
    ("футбол", ["футбол", "football", "sport"]),
    ("музык", ["музыка", "music", "song", "песня"]),
    ("песн", ["музыка", "music", "song"]),
    ("танц", ["танцы", "dance", "trend"]),
    ("еда", ["еда", "food", "recipe", "готовка"]),
    ("готов", ["готовка", "food", "recipe", "cooking"]),
    ("путешест", ["путешествия", "travel", "vlog"]),
    ("животн", ["животные", "pets", "cute", "cats", "dogs"]),
    ("кошк", ["котики", "cats", "pets", "cute"]),
    ("собак", ["собаки", "dogs", "pets", "cute"]),
    ("дет", ["дети", "kids", "family"]),
    ("семь", ["семья", "family", "mom", "dad"]),
    ("беремен", ["беременность", "mom", "baby", "family"]),
    ("свадьб", ["свадьба", "wedding", "love"]),
    ("красот", ["красота", "beauty", "makeup", "skincare"]),
    ("макияж", ["макияж", "makeup", "beauty"]),
    ("мода", ["мода", "fashion", "style", "outfit"]),
    ("авто", ["авто", "cars", "car", "drive"]),
    ("машина", ["авто", "cars", "car"]),
    ("технолог", ["технологии", "tech", "gadget"]),
    ("телефон", ["технологии", "tech", "smartphone"]),
    ("космос", ["космос", "space", "universe"]),
    ("истор", ["история", "history", "facts"]),
    ("факт", ["факты", "facts", "знания", "knowledge"]),
    ("совет", ["советы", "lifehack", "tips", "hack"]),
    ("лайфхак", ["лайфхак", "lifehack", "tips"]),
    ("реакц", ["реакция", "reaction", "react"]),
    ("обзор", ["обзор", "review", "разбор"]),
    ("разбор", ["разбор", "review", "аналитика"]),
    # универсальные «виральные» маркеры
    ("ого", ["шок", "wow", "amazing"]),
    ("вау", ["шок", "wow", "amazing"]),
    ("шок", ["шок", "wow", "amazing", "unbelievable"]),
    ("офиг", ["шок", "wow", "insane"]),
    ("жесть", ["шок", "crazy", "insane"]),
    ("топ", ["топ", "best", "top"]),
    ("лучш", ["лучшее", "best", "top"]),
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
    """Эвристика без LLM: популярные теги по теме текста + частые слова."""
    low = (text or "").lower()
    tags: List[str] = []
    # готовые популярные теги по теме
    for needle, tags_list in _TOPIC_TAGS:
        if needle in low:
            for t in tags_list:
                if t not in tags:
                    tags.append(t)
        if len(tags) >= limit:
            break
    # недобор — частые осмысленные слова (длина >= 4)
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
    for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True):
        if w not in tags:
            tags.append(w)
        if len(tags) >= limit:
            break
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
    # итоговый лимит — TikTok разрешает до 30 тегов на видео
    return out[:30]


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


def _extract_hook_fallback(text: str, limit: int = 70) -> str:
    """Без LLM: берём самую «интересную» короткую фразу (вопрос/интрига)."""
    import re

    sentences = re.split(r"[.!?…]+", (text or "").replace("\n", " "))
    sentences = [s.strip(" -\t") for s in sentences if len(s.strip()) >= 4]
    if not sentences:
        return (text or "").strip()[:limit]
    # приоритет — короткие фразы с «удержанием» (вопросы/интрига)
    candidates = []
    for needle in ("кто", "почему", "зачем", "что", "где", "как", "?", "смех", "ха",
                   "lol", "ого", "вау", "шок", "ужас", "правда", "тайна", "загадк",
                   "отрав", "убийств", "пропал", "нашёл", "секрет", "разгадк"):
        for sent in sentences:
            if needle in sent.lower():
                candidates.append(sent)
    if candidates:
        # самая короткая из подходящих
        return min(candidates, key=len)[:limit]
    if sentences:
        return min(sentences, key=len)[:limit]
    return (text or "").strip()[:limit]


# Эмодзи под настроение/тематику (добавляется в конец крючка)
_EMOJI_BY_TOPIC = [
    ("смех", "😂"), ("ха", "😂"), ("lol", "😂"), ("рж", "😂"),
    ("любов", "❤️"), ("поцел", "😘"), ("свадьб", "💍"),
    ("драм", "😱"), ("ужас", "😱"), ("страх", "😱"), ("кров", "🩸"),
    ("огонь", "🔥"), ("пожар", "🔥"), ("взрыв", "💥"), ("бум", "💥"),
    ("хими", "🧪"), ("лаборатор", "🧪"), ("наука", "🔬"),
    ("детектив", "🕵️"), ("убийств", "🕵️"), ("тайна", "🕵️"), ("загадк", "❓"),
    ("суд", "⚖️"), ("полиц", "🚔"), ("преступл", "🚔"),
    ("деньги", "💰"), ("богат", "💰"), ("золот", "💰"),
    ("спорт", "🏆"), ("футбол", "⚽"), ("бой", "🥊"),
    ("еда", "🍔"), ("готов", "🍳"),
    ("музык", "🎵"), ("песн", "🎶"), ("танц", "💃"),
    ("игра", "🎮"), ("космос", "🚀"), ("природ", "🌿"),
]
_DEFAULT_EMOJI = "🔥"


def _pick_emoji(text: str) -> str:
    low = (text or "").lower()
    for needle, emoji in _EMOJI_BY_TOPIC:
        if needle in low:
            return emoji
    return _DEFAULT_EMOJI


def generate_hook(
    text: str,
    llm_model: Optional[str] = None,
    llm_url: Optional[str] = None,
    limit: int = 60,
) -> str:
    """Короткий «крючок»: пара слов-интрига + эмодзи в конце.

    Формат примерно как у вас: «Кто отравил пробирку?🧪🔍». Сначала пробуем
    локальную LLM (просим 2-4 слова + эмодзи), иначе — эвристика по тексту.
    Пустой текст → пустая строка.
    """
    text = (text or "").strip()
    if not text:
        return ""
    emoji = _pick_emoji(text)
    if llm_model:
        try:
            import ollama
        except ImportError:
            return _extract_hook_fallback(text, limit) + emoji
        client = ollama.Client(host=llm_url) if llm_url else ollama
        prompt = (
            "Ты — топовый редактор TikTok-шортсов. Придумай ОДИН цепляющий крючок "
            "по тексту видео, строго по его ТЕМЕ/сюжету (не общие фразы): "
            "короткий вопрос, интрига или неожиданный факт из этого куска, 2-5 слов, "
            "чтобы зритель обязательно досмотрел до конца. "
            f"Строго до {limit} символов, на языке текста, без хештегов и кавычек. "
            "В конце поставь 1-2 уместных эмодзи.\n"
            f"Текст: {text!r}\nКрючок:"
        )
        try:
            resp = client.chat(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.6, "num_predict": 50},
            )
            hook = resp["message"]["content"].strip().strip('"').strip("'").rstrip(". ")
            hook = hook.splitlines()[0] if hook else ""
            if hook:
                # гарантируем наличие эмодзи
                if not any(ord(c) > 0x1F000 for c in hook):
                    hook = hook + " " + emoji
                return hook[:limit + 6]
        except Exception:
            pass
    return _extract_hook_fallback(text, limit) + emoji
