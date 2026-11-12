"""Azure text-to-speech (Hebrew, Arabic and many more), streamed as raw 8 kHz 16-bit PCM.

Like the Aura adapter, this asks the service for exactly the format the phone
line uses (``raw-8khz-16bit-mono-pcm``), so audio goes from Azure to the
caller without resampling. It uses Azure's plain REST endpoint through the
shared ``httpx`` client: the response is streamed in chunks, and stopping to
read it (the caller interrupted) closes the request, which ends the synthesis.

Built for the **free (F0) tier**, which allows 20 requests per minute and
500,000 characters per month:

* **No keep-warm pings** (``keep_warm_url`` is None). The connection is simply
  reused while it stays open; when Azure has dropped it, the next request
  reconnects and the extra time is logged.
* **One request per reply** (``one_request_per_reply``): the agent sends the
  whole reply, not sentence by sentence.
* **Fixed phrases cost nothing**: greeting, closings and so on are cached on
  disk per voice by :class:`~app.phrase_cache.PhraseCache`.
* **A rate limiter on our side** (:class:`RateLimiter`) refuses the 21st
  request of a minute before it is sent, and an HTTP 429 from Azure is turned
  into the same :class:`~app.speech.TTSRateLimited`, so the agent can say "one
  moment" and try once more.
* **Usage is counted** (:class:`UsageMeter`): requests and characters, per
  process and per calendar month in a small file, to watch the monthly quota.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional
from xml.sax.saxutils import escape

import httpx

from .speech import TTSError, TTSRateLimited

log = logging.getLogger("voice_agent.tts")

OUTPUT_FORMAT = "raw-8khz-16bit-mono-pcm"
#: Voices used when a profile names none.
DEFAULT_VOICES = {"he": "he-IL-HilaNeural", "ar": "ar-JO-SanaNeural", "en": "en-US-AvaNeural"}
FREE_TIER_REQUESTS_PER_MINUTE = 20
FREE_TIER_CHARS_PER_MONTH = 500_000
#: What to tell the agent to wait when Azure says 429 without a Retry-After header.
DEFAULT_RETRY_AFTER_S = 3.0


def endpoint(region: str) -> str:
    return f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"


def default_voice(language: str) -> Optional[str]:
    return DEFAULT_VOICES.get(language.split("-")[0].lower())


def build_ssml(voice: str, text: str) -> str:
    """The request body. ``text`` is escaped, so it can never add markup of its own."""
    locale = "-".join(voice.split("-")[:2])
    return (
        f"<speak version='1.0' xml:lang='{locale}'>"
        f"<voice name='{escape(voice, {chr(39): '&apos;'})}'>{escape(text)}</voice>"
        "</speak>"
    )


class RateLimiter:
    """At most ``limit`` requests in any ``window_s`` seconds (a sliding window)."""

    def __init__(self, limit: int = FREE_TIER_REQUESTS_PER_MINUTE, window_s: float = 60.0, now: Callable[[], float] = time.monotonic) -> None:
        self.limit = limit
        self.window_s = window_s
        self._now = now
        self._sent: deque[float] = deque()

    def _expire(self) -> None:
        cutoff = self._now() - self.window_s
        while self._sent and self._sent[0] <= cutoff:
            self._sent.popleft()

    def reserve(self) -> float:
        """Take a slot. Returns 0.0 if the request may go now, else how many seconds until one frees up."""
        self._expire()
        if self.limit <= 0 or len(self._sent) < self.limit:
            self._sent.append(self._now())
            return 0.0
        return max(0.001, self._sent[0] + self.window_s - self._now())

    @property
    def used(self) -> int:
        self._expire()
        return len(self._sent)


class UsageMeter:
    """Counts requests and characters, in total for this process and per calendar month on disk."""

    def __init__(self, directory: Optional[Path] = None, now: Callable[[], float] = time.time) -> None:
        self._dir = directory
        self._now = now
        self.requests = 0
        self.chars = 0

    def _month(self) -> str:
        return datetime.fromtimestamp(self._now(), timezone.utc).strftime("%Y-%m")

    def _path(self) -> Optional[Path]:
        return None if self._dir is None else self._dir / f"azure_tts_{self._month()}.json"

    def month(self) -> dict[str, Any]:
        """This month's totals: ``{"month", "requests", "chars"}``."""
        totals: dict[str, Any] = {"month": self._month(), "requests": 0, "chars": 0}
        path = self._path()
        if path is not None:
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                totals["requests"] = int(stored.get("requests", 0))
                totals["chars"] = int(stored.get("chars", 0))
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        return totals

    def add(self, chars: int) -> None:
        """One request for ``chars`` characters was sent. Never raises."""
        self.requests += 1
        self.chars += chars
        path = self._path()
        if path is None:
            return
        totals = self.month()
        totals["requests"] += 1
        totals["chars"] += chars
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(totals, handle)
            os.replace(tmp, path)
        except OSError as exc:
            log.warning("could not store TTS usage in %s: %s", path, exc)


class ConnectionWatch:
    """Notices when a request had to open a new connection, and how long that took.

    httpx reports low-level steps through a "trace" callback. A request on a
    reused connection has no connect / TLS steps at all; a fresh one has both.
    """

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self._started: Optional[float] = None
        self.connect_ms: Optional[float] = None  # None = the connection was reused

    async def __call__(self, event: str, info: dict[str, Any]) -> None:
        if event == "connection.connect_tcp.started":
            self._started = self._now()
        elif event in ("connection.start_tls.complete", "connection.connect_tcp.complete") and self._started is not None:
            self.connect_ms = (self._now() - self._started) * 1000.0


class AzureTTS:
    """:class:`~app.speech.TTSProvider` for Azure AI Speech (REST, streamed)."""

    name = "Azure"
    #: Free tier: every request counts, so the server must not ping this service.
    keep_warm_url: Optional[str] = None
    #: The agent sends a whole reply in one request instead of one per sentence.
    one_request_per_reply = True

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        region: str,
        voice: str,
        limiter: Optional[RateLimiter] = None,
        usage: Optional[UsageMeter] = None,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._url = endpoint(region)
        self.voice = voice
        self.limiter = limiter if limiter is not None else RateLimiter()
        self.usage = usage if usage is not None else UsageMeter()
        self.reconnects = 0
        self._requests_sent = 0

    @property
    def cache_key(self) -> str:
        return f"azure:{self.voice}"

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield PCM chunks for ``text`` as Azure generates them.

        Raises :class:`TTSRateLimited` (before or instead of any audio) when the
        per-minute limit is reached, :class:`TTSError` for anything else.
        """
        wait_s = self.limiter.reserve()
        if wait_s > 0:
            log.warning(
                "Azure TTS: own limit of %d requests per %.0f s reached; next slot in %.1f s",
                self.limiter.limit, self.limiter.window_s, wait_s,
            )
            raise TTSRateLimited(f"Azure request limit reached ({self.limiter.limit} per minute)", wait_s)
        watch = ConnectionWatch()
        self._requests_sent += 1
        try:
            async with self._client.stream(
                "POST",
                self._url,
                content=build_ssml(self.voice, text).encode("utf-8"),
                headers={
                    "Ocp-Apim-Subscription-Key": self._api_key,
                    "Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": OUTPUT_FORMAT,
                    "User-Agent": "voice-agent",
                },
                extensions={"trace": watch},
            ) as response:
                if watch.connect_ms is not None and self._requests_sent > 1:
                    # Expected now and then: Azure closes idle connections and we do not ping.
                    self.reconnects += 1
                    log.info("Azure TTS: idle connection was gone; reconnecting cost %.0f ms", watch.connect_ms)
                if response.status_code == 429:
                    await response.aread()
                    retry_after = _retry_after(response.headers.get("retry-after"))
                    log.warning("Azure TTS: HTTP 429 (too many requests); retry after %.1f s", retry_after)
                    raise TTSRateLimited("Azure HTTP 429", retry_after)
                if response.status_code != 200:
                    await response.aread()
                    raise TTSError(f"Azure HTTP {response.status_code}")
                self.usage.add(len(text))  # accepted: these characters count towards the monthly quota
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.HTTPError as exc:
            raise TTSError(f"Azure request failed: {type(exc).__name__}") from None

    async def synthesize(self, text: str) -> bytes:
        """Whole utterance as one bytes object (for phrases we cache)."""
        return b"".join([chunk async for chunk in self.stream(text)])


def _retry_after(header: Optional[str]) -> float:
    try:
        value = float(header) if header else DEFAULT_RETRY_AFTER_S
    except ValueError:
        value = DEFAULT_RETRY_AFTER_S
    return min(max(value, 0.5), 60.0)
