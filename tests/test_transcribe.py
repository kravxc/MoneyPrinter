import os
import wave

from moneyprinter.transcribe import _make_chunks, _merge_chunk_results


def _write_test_wav(path, seconds=3, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * (rate * seconds))


def test_make_chunks_covers_full_duration(tmp_path):
    wav = tmp_path / "t.wav"
    _write_test_wav(wav)
    chunks = _make_chunks(str(wav), str(tmp_path), chunk_sec=1.0, overlap_sec=0.2)
    assert chunks
    assert len(chunks) >= 3
    # покрытие от начала до конца
    assert chunks[0][1] == 0.0
    last = chunks[-1]
    assert last[1] + last[2] >= 3.0


def test_make_chunks_writes_valid_wavs(tmp_path):
    wav = tmp_path / "t.wav"
    _write_test_wav(wav)
    chunks = _make_chunks(str(wav), str(tmp_path), chunk_sec=1.0, overlap_sec=0.2)
    for path, _, _ in chunks:
        with wave.open(path, "rb") as w:
            assert w.getnframes() > 0


def test_merge_chunk_results_dedups_overlap():
    # чанк 1 покрывает [0, 3], чанк 2 начинается на 2.8 (перекрытие)
    chunks = [
        ("chunk0.wav", 0.0, 3.0),
        ("chunk1.wav", 2.8, 3.0),
    ]
    results = [
        [(0.5, 1.0, "первая", 0.0), (1.5, 2.9, "вторая", 0.0)],
        # в зоне перекрытия (старт < 2.8+0.2) — должно быть отброшено
        [(2.9, 3.5, "дубликат", 0.0), (3.6, 4.0, "хвост", 0.0)],
    ]
    merged = _merge_chunk_results(results, chunks, overlap_sec=0.2)
    texts = [s.text for s in merged]
    assert "первая" in texts
    assert "вторая" in texts
    assert "дубликат" not in texts
    assert "хвост" in texts


def test_merge_chunk_results_first_chunk_keeps_all():
    chunks = [("c0.wav", 0.0, 3.0)]
    results = [[(0.1, 1.0, "текст", 0.0)]]
    merged = _merge_chunk_results(results, chunks)
    assert len(merged) == 1
    assert merged[0].text == "текст"