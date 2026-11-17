"""scripts/stt_sentences.py: cutting a test call into sentences and scoring transcripts (no network)."""

from __future__ import annotations

import math
import struct
from pathlib import Path

from scripts.stt_sentences import normalise, piece, read_sentences, similarity, split_sentences

RATE = 8000


def tone(seconds: float, amplitude: int = 6000) -> bytes:
    return b"".join(struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 300 * n / RATE))) for n in range(int(seconds * RATE)))


def hiss(seconds: float) -> bytes:
    return b"".join(struct.pack("<h", 40 if n % 2 else -40) for n in range(int(seconds * RATE)))


def test_recording_is_cut_at_the_pauses() -> None:
    audio = hiss(1.0) + tone(1.5) + hiss(2.0) + tone(0.6) + hiss(0.4) + tone(0.6) + hiss(2.0) + tone(0.1) + hiss(1.5) + tone(1.0) + hiss(1.0)
    spans = split_sentences(audio)
    assert len(spans) == 3  # the short pause (0.4 s) does not split; the 0.1 s click is not a sentence
    (a0, a1), (b0, b1), (c0, c1) = spans
    assert abs(a0 - 0.7) < 0.1 and abs(a1 - 2.8) < 0.1  # 1.0-2.5 s plus 0.3 s on each side
    assert abs(b0 - 4.2) < 0.1 and abs(b1 - 6.4) < 0.1
    assert c1 <= len(audio) / 2 / RATE
    assert len(piece(audio, a0, a1)) == int(a1 * RATE) * 2 - int(a0 * RATE) * 2


def test_silence_gives_no_pieces() -> None:
    assert split_sentences(hiss(3.0)) == [] and split_sentences(b"") == []


def test_similarity_ignores_punctuation_and_spelling_variants() -> None:
    assert similarity("لا، أول مرة.", "لا اول مره") == 1.0
    assert normalise("شو ساعات الدوام عندكم؟") == "شو ساعات الدوام عندكم"
    assert similarity("أول مرة.", "اونجا راند") < 0.5
    assert similarity("أول مرة.", "") == 0.0


def test_the_twelve_sentences_are_shipped() -> None:
    sentences = read_sentences(Path(__file__).resolve().parent.parent / "scripts" / "stt_sentences_ar.txt")
    assert len(sentences) == 12 and sentences[1] == "أول مرة." and "תור" in sentences[6]
