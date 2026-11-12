"""Audio for fixed phrases (greeting, apology, closings): memory, then disk, then TTS.

Fixed phrases are synthesized once and reused. That makes the greeting start
instantly, costs nothing per call, and means the apology still works when the
TTS service is the thing that just failed. Files are keyed by the provider's
``cache_key`` (its voice) + text, so changing either produces a new file.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from .audiosocket import SAMPLE_RATE_HZ
from .speech import TTSError, TTSProvider

log = logging.getLogger("voice_agent.tts")


class PhraseCache:
    def __init__(self, tts: TTSProvider, directory: Path) -> None:
        self._tts = tts
        self._dir = directory
        self._memory: dict[str, bytes] = {}
        self._lock = asyncio.Lock()
        self.chars_synthesized = 0

    def _path(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self._tts.cache_key}|{SAMPLE_RATE_HZ}|{text}".encode()).hexdigest()[:32]
        return self._dir / f"{digest}.pcm"

    def cached(self, text: str) -> Optional[bytes]:
        """Audio if it is already in memory (never does I/O or network)."""
        return self._memory.get(text)

    async def get(self, text: str) -> bytes:
        """Audio for ``text``; synthesizes on first use. Raises :class:`TTSError`."""
        audio = self._memory.get(text)
        if audio is not None:
            return audio
        async with self._lock:  # one synthesis at a time; later callers find it in memory
            return await self._load_or_synthesize(text)

    async def _load_or_synthesize(self, text: str) -> bytes:
        audio = self._memory.get(text)
        if audio is not None:
            return audio
        path = self._path(text)
        try:
            audio = path.read_bytes()
        except OSError:
            audio = b""
        if not audio:
            audio = await self._tts.synthesize(text)
            if not audio:
                raise TTSError(f"{self._tts.name} returned no audio")
            self.chars_synthesized += len(text)
            try:
                self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(audio)
                os.replace(tmp, path)
            except OSError as exc:
                log.warning("could not store phrase audio in %s: %s", self._dir, exc)
        self._memory[text] = audio
        return audio

    async def warm(self, texts: list[str]) -> None:
        """Best-effort pre-generation at start-up. Never raises."""
        for text in texts:
            try:
                await self.get(text)
            except TTSError as exc:
                log.warning("could not pre-generate phrase audio (%s); will retry on first use", exc)
