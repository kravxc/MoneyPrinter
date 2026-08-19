import numpy as np

from moneyprinter.audio import burstiness, energy_score_window, loudness_envelope, silence_gaps


def test_loudness_envelope_shape():
    x = np.zeros(16000 * 2)
    rms, times = loudness_envelope(x, 16000, win_sec=1.0)
    assert len(rms) == 2
    assert np.allclose(rms, 0.0)


def test_loudness_detects_quiet_and_loud():
    x = np.zeros(16000 * 4)
    x[16000:32000] = 0.5
    rms, _ = loudness_envelope(x, 16000, win_sec=1.0)
    assert rms[0] < 0.01
    assert rms[1] > 0.4


def test_silence_gaps():
    x = np.zeros(16000 * 5)
    x[1000:4000] = 0.5
    gaps = silence_gaps(x, 16000, threshold_db=-40, min_silence_sec=1.0)
    assert len(gaps) >= 1
    for start, end in gaps:
        assert end - start >= 1.0


def test_energy_score_in_range():
    x = np.zeros(16000 * 3)
    x[1000:2000] = 0.3
    score, _ = energy_score_window(x, 16000, win_sec=0.5)
    assert score.min() >= 0.0
    assert score.max() <= 1.0 + 1e-9


def test_burstiness_quiet_is_zero():
    x = np.zeros(16000 * 2)
    assert np.allclose(burstiness(x, 16000, 1.0), 0.0)