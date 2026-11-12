"""Deepgram Aura text-to-speech, streamed as raw 8 kHz 16-bit PCM.

We ask Aura for exactly the format AudioSocket wants (``linear16``, 8000 Hz,
no container), so audio goes from Deepgram to the caller without resampling.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

import httpx

from .audiosocket import SAMPLE_RATE_HZ
from .phrase_cache import PhraseCache  # noqa: F401  (kept importable from here)
from .speech import TTSError

log = logging.getLogger("voice_agent.tts")

AURA_URL = "https://api.deepgram.com/v1/speak"


class AuraTTS:
    """:class:`~app.speech.TTSProvider` for Deepgram Aura (English voices)."""

    name = "Aura"
    #: Deepgram drops idle HTTPS connections after ~5 s, so the server touches
    #: this URL every few seconds during a call (no credentials are sent).
    keep_warm_url: Optional[str] = "https://api.deepgram.com/"

    def __init__(self, client: httpx.AsyncClient, api_key: str, model: str = "aura-2-thalia-en") -> None:
        self._client = client
        self._api_key = api_key
        self.model = model

    @property
    def voice(self) -> str:
        return self.model

    @property
    def cache_key(self) -> str:
        return self.model  # unchanged from before the adapter refactor: existing cache files stay valid

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
