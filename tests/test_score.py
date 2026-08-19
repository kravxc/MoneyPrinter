import numpy as np

from moneyprinter.models import ClipCandidate, SceneBreak, TimestampedText
from moneyprinter.score import (
    generate_candidates,
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