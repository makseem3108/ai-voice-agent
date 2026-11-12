"""Deepgram Aura text-to-speech, streamed as raw 8 kHz 16-bit PCM.

We ask Aura for exactly the format AudioSocket wants (``linear16``, 8000 Hz,
no container), so audio goes from Deepgram to the caller without resampling.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx

from .audiosocket import SAMPLE_RATE_HZ

log = logging.getLogger("voice_agent.tts")

AURA_URL = "https://api.deepgram.com/v1/speak"


class TTSError(Exception):
    """Speech synthesis failed. Message is safe to log."""


class AuraTTS:
    def __init__(self, client: httpx.AsyncClient, api_key: str, model: str = "aura-2-thalia-en") -> None:
        self._client = client
        self._api_key = api_key
        self.model = model

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield PCM chunks for ``text`` as Aura generates them."""
        params = {
            "model": self.model,
            "encoding": "linear16",
            "sample_rate": str(SAMPLE_RATE_HZ),
            "container": "none",
        }
        try:
            async with self._client.stream(
                "POST",
                AURA_URL,
                params=params,
                json={"text": text},
                headers={"Authorization": f"Token {self._api_key}"},
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise TTSError(f"Aura HTTP {response.status_code}")
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.HTTPError as exc:
            raise TTSError(f"Aura request failed: {type(exc).__name__}") from None

    async def synthesize(self, text: str) -> bytes:
        """Whole utterance as one bytes object (for phrases we cache)."""
        return b"".join([chunk async for chunk in self.stream(text)])


class PhraseCache:
    """Audio for fixed phrases (greeting, apology, goodbye): memory, then disk, then Aura.

    Fixed phrases are synthesized once and reused, which makes the greeting
    start instantly and means the apology still works when Aura is the thing
    that just failed. Files are keyed by voice model + text, so changing either
    produces a new file.
    """

    def __init__(self, tts: AuraTTS, directory: Path) -> None:
        self._tts = tts
        self._dir = directory
        self._memory: dict[str, bytes] = {}
        self._lock = asyncio.Lock()
        self.chars_synthesized = 0

    def _path(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self._tts.model}|{SAMPLE_RATE_HZ}|{text}".encode()).hexdigest()[:32]
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
                raise TTSError("Aura returned no audio")
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
