"""Local voice detection for fast barge-in.

Speech-to-text tells us reliably *that* the caller started a turn, but it can
take a second or two to say so (it waits until it recognises words). That is
too slow for "stop talking when I interrupt". So while the bot is speaking we
also watch the caller's audio ourselves and pause the bot as soon as it looks
like they are talking; speech-to-text then confirms or the bot resumes.

Two checks must both pass for a 20 ms frame to count as caller speech:

* **WebRTC VAD** (Google's voice activity detector, via ``webrtcvad``): a small,
  battle-tested classifier that separates voice from noise far better than a
  plain volume threshold.
* **Loudness gate**: the frame's RMS level must exceed ``min_rms``. The bot's
  own voice leaking back through the caller's phone (echo) *is* voice, so the
  VAD alone would accept it - but echo is much quieter than someone talking
  into the microphone. Measured on our own TTS audio: a copy 20 dB quieter
  still looks like voice to the VAD on 66 % of frames, but its RMS stays under
  ~370, while direct speech is typically 800-3700.

The detector fires when enough frames inside a short sliding window qualify.
"""

from __future__ import annotations

import array
import math
from collections import deque
from typing import Callable, Optional

from .audiosocket import SAMPLE_RATE_HZ

FRAME_BYTES = 320  # 20 ms at 8 kHz, 16-bit - a frame size WebRTC VAD accepts
FRAME_MS = 20


def frame_rms(frame: bytes) -> float:
    """Root-mean-square level of a 16-bit little-endian PCM frame (0..32768)."""
    samples = array.array("h")
    samples.frombytes(frame[: len(frame) - (len(frame) % 2)])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def webrtc_classifier(mode: int = 3) -> Callable[[bytes], bool]:
    """Voice/non-voice classifier backed by WebRTC VAD (mode 0-3; 3 = strictest)."""
    import webrtcvad  # imported lazily so tests can run without it

    vad = webrtcvad.Vad(mode)
    return lambda frame: bool(vad.is_speech(frame, SAMPLE_RATE_HZ))


class BargeInDetector:
    """Says "the caller is talking" after ``min_speech_ms`` of loud voice in ``window_ms``."""

    def __init__(
        self,
        *,
        min_rms: float = 500.0,
        min_speech_ms: int = 120,
        window_ms: int = 200,
        is_speech: Optional[Callable[[bytes], bool]] = None,
        mode: int = 3,
    ) -> None:
        self._min_rms = min_rms
        self._needed = max(1, math.ceil(min_speech_ms / FRAME_MS))
        self._window: deque[bool] = deque(maxlen=max(self._needed, math.ceil(window_ms / FRAME_MS)))
        self._is_speech = is_speech if is_speech is not None else webrtc_classifier(mode)
        self.last_rms = 0.0

    def reset(self) -> None:
        self._window.clear()

    def feed(self, frame: bytes) -> bool:
        """Add one 20 ms frame; True when the caller appears to be speaking."""
        if len(frame) != FRAME_BYTES:
            return False
        self.last_rms = frame_rms(frame)
        voiced = self.last_rms >= self._min_rms and self._is_speech(frame)
        self._window.append(voiced)
        return sum(self._window) >= self._needed
