import numpy as np

from moneyprinter.models import ClipCandidate, SceneBreak, TimestampedText
from moneyprinter.score import (
    _trim_banner_overlap,
    generate_candidates,
    group_segments,
    non_max_suppress,
    score_text_segment,
)


def _seg(start, end, text, nsp=0.0):
    return TimestampedText(start=start, end=end, text=text, no_speech_prob=nsp)


def test_score_laughter_high():
    a = score_text_segment(_seg(0, 1, "обычная реплика"))
    b = score_text_segment(_seg(0, 1, "ХАХАХА LOL ОМГ круто!!"))
    assert b > a


def test_score_stop_words_penalized():
    a = score_text_segment(_seg(0, 1, "wow ничего себе"))
    b = score_text_segment(_seg(0, 1, "это реклама и музыка играет"))
    assert a > b


def test_score_caps_bonus():
    a = score_text_segment(_seg(0, 1, "смотри что я нашёл"))
    b = score_text_segment(_seg(0, 1, "СМОТРИ ЧТО Я НАШЁЛ"))
    assert b > a


def test_generate_candidates_bounds():
    segs = [_seg(10, 14, "какой смешной момент хаха")]
    scenes = [SceneBreak(time=8.0), SceneBreak(time=20.0)]
    silences = [(9.0, 10.0), (14.0, 15.0)]
    e = np.ones(100)
    t = np.linspace(0, 30, 100)
    cands = generate_candidates(e, t, segs, scenes, silences, duration=30.0)
    assert len(cands) == 1
    c = cands[0]
    assert c.start >= 0 and c.end <= 30
    assert c.total_score > 0


def test_group_segments_merges_close_breaks_far():
    segs = [
        _seg(0, 2, "реплика один"),
        _seg(2.3, 4.5, "реплика два"),
        _seg(8.0, 10.0, "далёкая реплика"),
    ]
    groups = group_segments(segs, scene_breaks=[], max_gap=2.0)
    assert len(groups) == 2
    assert len(groups[0]) == 2
    assert len(groups[1]) == 1


def test_group_segments_respects_scene_break():
    segs = [
        _seg(0, 2, "до смены сцены"),
        _seg(2.5, 4, "после смены сцены"),
    ]
    groups = group_segments(segs, scene_breaks=[SceneBreak(time=2.3)], max_gap=5.0)
    assert len(groups) == 2


def test_generate_candidates_splits_long_story():
    # история длиннее max_duration делится на последовательные части без потерь
    segs = [_seg(0, 40, "длинная история"), _seg(40.5, 80, "продолжение")]
    e = np.ones(100)
    t = np.linspace(0, 100, 100)
    cands = generate_candidates(
        e, t, segs, [], [], duration=100.0, min_duration=4.0, max_duration=50.0
    )
    assert len(cands) == 2
    assert cands[0].start == 0
    assert cands[1].end >= 80


def test_generate_candidates_groups_into_story():
    segs = [
        _seg(0, 2, "обычная реплика"),
        _seg(2.4, 4.5, "и продолжение истории"),
    ]
    e = np.ones(100)
    t = np.linspace(0, 10, 100)
    cands = generate_candidates(e, t, segs, [], [], duration=10.0, min_duration=4.0)
    assert len(cands) == 1
    assert cands[0].text == "обычная реплика и продолжение истории"


def test_group_segments_splits_around_ads():
    segs = [
        _seg(0, 2, "первая история"),
        _seg(2.3, 4, "рекламный банер"),
        _seg(4.3, 7, "вторая история"),
    ]
    segs[1].is_ad = True
    groups = group_segments(segs, scene_breaks=[], max_gap=5.0, remove_ads=True)
    assert len(groups) == 2
    assert len(groups[0]) == 1 and groups[0][0].text == "первая история"
    assert len(groups[1]) == 1 and groups[1][0].text == "вторая история"


def test_group_segments_keeps_ads_if_requested():
    segs = [
        _seg(0, 2, "первая"),
        _seg(2.3, 4, "банер"),
    ]
    segs[1].is_ad = True
    groups = group_segments(segs, scene_breaks=[], max_gap=5.0, remove_ads=False)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_generate_candidates_skips_ad_stories():
    segs = [
        _seg(0, 2, "обычная реплика"),
        _seg(2.4, 5, "рекламный банер на экране"),
    ]
    segs[1].is_ad = True
    e = np.ones(50)
    t = np.linspace(0, 6, 50)
    cands = generate_candidates(e, t, segs, [], [], duration=6.0, remove_ads=True)
    assert len(cands) == 1
    assert "банер" not in cands[0].text


def test_trim_banner_overlap_shifts_window_right_keeping_length():
    # окно 0-20, банер 8-15, хватает места → сдвиг вправо, длина сохраняется
    r = _trim_banner_overlap(0.0, 20.0, [(8.0, 15.0)], duration=50.0, min_duration=4.0)
    assert r == (15.0, 35.0)


def test_trim_banner_overlap_shifts_window_left():
    # окно 20-50, банер 45-60 → сдвиг влево, окно 15-45 (ближе к исходному)
    r = _trim_banner_overlap(20.0, 50.0, [(45.0, 60.0)], duration=100.0, min_duration=4.0)
    assert r == (15.0, 45.0)


def test_trim_banner_overlap_no_overlap_unchanged():
    r = _trim_banner_overlap(0.0, 10.0, [(20.0, 30.0)], duration=60.0, min_duration=4.0)
    assert r == (0.0, 10.0)


def test_trim_banner_overlap_too_short_dropped():
    # справа нет места, слева остаётся слишком мало → None
    r = _trim_banner_overlap(0.0, 10.0, [(2.0, 20.0)], duration=15.0, min_duration=15.0)
    assert r is None


def test_generate_candidates_trims_banner_overlap():
    segs = [_seg(0, 3, "реплика"), _seg(3.2, 5, "ещё")]
    e = np.ones(100)
    t = np.linspace(0, 10, 100)
    cands = generate_candidates(
        e, t, segs, [], [], duration=10.0, min_duration=4.0,
        remove_ads=True, banner_ranges=[(3.0, 10.0)],
    )
    assert len(cands) == 1
    assert cands[0].end == 3.0


def test_non_max_suppress_overlap():
    a = ClipCandidate(start=0, end=10, energy_score=1.0)
    b = ClipCandidate(start=5, end=15, energy_score=0.1)
    picked = non_max_suppress([a, b])
    assert len(picked) == 1
    assert picked[0] is a


def test_non_max_suppress_disjoint():
    a = ClipCandidate(start=0, end=5, energy_score=0.5)
    b = ClipCandidate(start=20, end=25, energy_score=0.5)
    picked = non_max_suppress([a, b])
    assert len(picked) == 2