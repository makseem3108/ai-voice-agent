"""Optional debug recording of the caller's audio (``DEBUG_RECORD_AUDIO=true``).

OFF by default and meant for test calls only: recordings are personal data.
Files go to ``data/recordings/<uuid>.wav`` (owner-only, git-ignored) and are
deleted automatically once they are older than :data:`RETENTION_DAYS`. The
clean-up runs even when recording is switched off again, so nothing lingers.
"""

from __future__ import annotations

import logging
import os
import time
import wave
from pathlib import Path
from typing import BinaryIO, Optional

from .audiosocket import BYTES_PER_SAMPLE, SAMPLE_RATE_HZ

log = logging.getLogger("voice_agent.recording")

RETENTION_DAYS = 7


def cleanup_old_recordings(directory: Path, *, max_age_days: float = RETENTION_DAYS, now: Optional[float] = None) -> int:
    """Delete ``*.wav`` files older than ``max_age_days``. Returns how many were removed."""
    cutoff = (time.time() if now is None else now) - max_age_days * 86400
    removed = 0
    try:
        candidates = list(directory.glob("*.wav"))
    except OSError:
        return 0
    for path in candidates:
        try:
            if path.is_file() and not path.is_symlink() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError as exc:
            log.warning("could not remove old recording %s: %s", path.name, exc)
    if removed:
        log.info("removed %d recording(s) older than %g days", removed, max_age_days)
    return removed


class InboundRecorder:
    """Writes the caller's audio of one call to a WAV file. Never raises into the call."""

    def __init__(self, directory: Path, call_id: str) -> None:
        self.path = directory / f"{call_id}.wav"
        self._wav: Optional[wave.Wave_write] = None
        self._handle: Optional[BinaryIO] = None
        try:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            self._handle = os.fdopen(fd, "wb")
            self._wav = wave.open(self._handle, "wb")
            self._wav.setnchannels(1)
            self._wav.setsampwidth(BYTES_PER_SAMPLE)
            self._wav.setframerate(SAMPLE_RATE_HZ)
        except (OSError, wave.Error) as exc:
            log.error("cannot record call audio to %s: %s", self.path, exc)
            self._wav = None

    def write(self, pcm: bytes) -> None:
        if self._wav is None:
            return
        try:
            self._wav.writeframesraw(pcm)
        except (OSError, wave.Error) as exc:
            log.error("recording failed, disabling: %s", exc)
            self.close()

    def close(self) -> None:
        if self._wav is not None:
            wav, self._wav = self._wav, None
            try:
                wav.close()  # patches the WAV header with the final length
            except (OSError, wave.Error):
                pass
        if self._handle is not None:
            handle, self._handle = self._handle, None
            try:
                handle.close()  # wave does not close a file object it was handed
            except OSError:
                pass
