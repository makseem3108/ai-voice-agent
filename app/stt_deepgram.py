"""Deepgram live speech-to-text over the raw streaming WebSocket API.

Three independent pieces:

* **Message parsing** (:func:`parse_message`) - pure, turns Deepgram JSON into
  small event dataclasses.
* **Timing** (:class:`AudioClock`, :class:`TranscriptTracker`) - pure, maps
  Deepgram's audio timestamps to our wall clock and computes STT latency.
* **I/O** (:class:`DeepgramStream`) - one WebSocket per call: audio out,
  events in. Never raises into the call: on any failure it logs, marks itself
  failed and silently drops further audio.

The API key travels only in the ``Authorization`` header, never in the URL,
and is never logged.
"""

from __future__ import annotations

import asyncio
import bisect
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol, Union
from urllib.parse import urlencode

from .audiosocket import SAMPLE_RATE_HZ, audio_duration_s

log = logging.getLogger("voice_agent.stt")

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"

#: Deepgram closes a stream that receives neither audio nor KeepAlive for ~10 s.
KEEPALIVE_AFTER_IDLE_S = 4.0
#: Audio buffered while the WebSocket is still connecting (250 frames = 5 s).
MAX_QUEUED_FRAMES = 250
CONNECT_TIMEOUT_S = 5.0
#: After CloseStream, how long to wait for Deepgram's last results.
CLOSE_GRACE_S = 3.0


# --------------------------------------------------------------------------- params
@dataclass(frozen=True)
class DeepgramParams:
    model: str = "nova-3"
    language: str = "en"
    endpointing_ms: int = 400
    utterance_end_ms: int = 1000

    def query(self) -> dict[str, str]:
        return {
            "model": self.model,
            "language": self.language,
            "encoding": "linear16",
            "sample_rate": str(SAMPLE_RATE_HZ),
            "channels": "1",
            "interim_results": "true",
            "endpointing": str(self.endpointing_ms),
            "utterance_end_ms": str(self.utterance_end_ms),
            "vad_events": "true",
            "smart_format": "true",
        }

    def url(self) -> str:
        """Request URL. Contains no credentials, so it is safe to log."""
        return f"{DEEPGRAM_URL}?{urlencode(self.query())}"


# --------------------------------------------------------------------------- events
@dataclass(frozen=True)
class Transcript:
    """A ``Results`` message: Deepgram's text for one slice of audio."""

    text: str
    is_final: bool  # this slice will not be revised any more
    speech_final: bool  # endpointing fired: the speaker paused
    start: float  # audio seconds from stream start
    duration: float
    confidence: float = 0.0
    last_word_end: Optional[float] = None  # audio time the last word ended

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class SpeechStarted:
    timestamp: float  # audio seconds


@dataclass(frozen=True)
class UtteranceEnd:
    last_word_end: float  # audio seconds


@dataclass(frozen=True)
class StreamMetadata:
    request_id: str
    duration: float  # audio seconds Deepgram processed


@dataclass(frozen=True)
class SttError:
    message: str


@dataclass(frozen=True)
class Unknown:
    type: str


SttEvent = Union[Transcript, SpeechStarted, UtteranceEnd, StreamMetadata, SttError, Unknown]


def _num(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def parse_message(raw: Union[str, bytes]) -> SttEvent:
    """Turn one Deepgram WebSocket message into an event. Never raises."""
    try:
        msg = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return SttError("unparseable message from Deepgram")
    if not isinstance(msg, dict):
        return SttError("unexpected message shape from Deepgram")

    kind = msg.get("type")
    if kind == "Results":
        alternatives = (msg.get("channel") or {}).get("alternatives") or [{}]
        best = alternatives[0] if isinstance(alternatives[0], dict) else {}
        words = best.get("words") or []
        last_end: Optional[float] = None
        if words and isinstance(words[-1], dict) and "end" in words[-1]:
            last_end = _num(words[-1]["end"])
        return Transcript(
            text=str(best.get("transcript") or "").strip(),
            is_final=bool(msg.get("is_final")),
            speech_final=bool(msg.get("speech_final")),
            start=_num(msg.get("start")),
            duration=_num(msg.get("duration")),
            confidence=_num(best.get("confidence")),
            last_word_end=last_end,
        )
    if kind == "SpeechStarted":
        return SpeechStarted(timestamp=_num(msg.get("timestamp")))
    if kind == "UtteranceEnd":
        return UtteranceEnd(last_word_end=_num(msg.get("last_word_end")))
    if kind == "Metadata":
        return StreamMetadata(
            request_id=str(msg.get("request_id") or ""), duration=_num(msg.get("duration"))
        )
    if kind == "Error":
        text = msg.get("description") or msg.get("message") or msg.get("reason") or "unknown error"
        return SttError(str(text))
    return Unknown(type=str(kind))


# --------------------------------------------------------------------------- timing
class AudioClock:
    """Maps a position in the audio stream to the wall-clock time we got that audio.

    Deepgram timestamps everything in *audio seconds since the stream began*.
    We know, for every frame we forwarded, how much audio came before it and
    the moment it arrived from Asterisk. So "the word ended at audio second
    7.36" can be translated into "that sound reached this server at 14:02:11.480".
    """

    def __init__(self) -> None:
        self._ends: list[float] = []  # cumulative audio seconds at the end of each frame
        self._walls: list[float] = []  # wall-clock arrival time of that frame
        self._total = 0.0

    @property
    def total_seconds(self) -> float:
        """Audio seconds sent so far (what Deepgram bills)."""
        return self._total

    def mark(self, frame_seconds: float, arrival_wall: float) -> None:
        self._total += frame_seconds
        self._ends.append(self._total)
        self._walls.append(arrival_wall)

    def wall_time_at(self, audio_t: float) -> Optional[float]:
        """Arrival time of the frame containing audio second ``audio_t``."""
        if not self._ends:
            return None
        # tiny epsilon so a timestamp exactly on a frame boundary maps to that frame
        i = bisect.bisect_left(self._ends, audio_t - 1e-9)
        return self._walls[min(i, len(self._walls) - 1)]

    def latency_ms(self, audio_t: float, arrival_wall: float) -> Optional[float]:
        """Milliseconds between hearing audio second ``audio_t`` and ``arrival_wall``."""
        heard = self.wall_time_at(audio_t)
        if heard is None:
            return None
        return max(0.0, (arrival_wall - heard) * 1000.0)


@dataclass
class TranscriptTracker:
    """Turns raw STT events into log records and per-call statistics.

    **STT latency** of a final transcript =
    (wall time the final arrived) - (wall time we received the audio in which
    the last word of that transcript ended, looked up via :class:`AudioClock`).
    It therefore contains Deepgram's wait-for-silence, its processing time and
    the network round trip - i.e. how long after the caller stopped talking we
    actually know what they said.
    """

    clock: AudioClock
    finals: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    request_id: str = ""
    deepgram_duration: Optional[float] = None
    _last_word_end: Optional[float] = None
    _utterance_open: bool = False

    def on_event(self, event: SttEvent, arrival_wall: float) -> list[dict[str, Any]]:
        """Process one event; return zero or more records to log / store."""
        if isinstance(event, Transcript):
            return self._on_transcript(event, arrival_wall)
        if isinstance(event, SpeechStarted):
            return [{"type": "speech_start", "audio_t": round(event.timestamp, 3)}]
        if isinstance(event, UtteranceEnd):
            self._utterance_open = False
            return [{"type": "utterance_end", "last_word_end": round(event.last_word_end, 3)}]
        if isinstance(event, StreamMetadata):
            self.request_id = event.request_id
            self.deepgram_duration = event.duration
            return []
        if isinstance(event, SttError):
            return [{"type": "stt_error", "message": event.message}]
        return []

    def _on_transcript(self, ev: Transcript, arrival_wall: float) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if ev.text:
            self._utterance_open = True
            word_end = ev.last_word_end if ev.last_word_end is not None else ev.end
            if not ev.is_final:
                records.append({"type": "interim", "text": ev.text, "audio_end": round(ev.end, 3)})
            else:
                self._last_word_end = word_end
                latency = self.clock.latency_ms(word_end, arrival_wall)
                self.finals += 1
                if latency is not None:
                    self.latencies_ms.append(latency)
                records.append(
                    {
                        "type": "final",
                        "text": ev.text,
                        "confidence": round(ev.confidence, 3),
                        "audio_start": round(ev.start, 3),
                        "last_word_end": round(word_end, 3),
                        "latency_ms": None if latency is None else round(latency, 1),
                        "speech_final": ev.speech_final,
                    }
                )
        if ev.speech_final and self._utterance_open and self._last_word_end is not None:
            self._utterance_open = False
            silence = max(0.0, ev.end - self._last_word_end)
            records.append({"type": "speech_end", "silence_s": round(silence, 2)})
        return records

    @property
    def avg_latency_ms(self) -> Optional[float]:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else None

    @property
    def max_latency_ms(self) -> Optional[float]:
        return max(self.latencies_ms) if self.latencies_ms else None


# --------------------------------------------------------------------------- I/O
class WebSocketLike(Protocol):
    """The part of a ``websockets`` connection we use (lets tests pass a fake)."""

    async def send(self, message: Union[str, bytes]) -> None: ...
    async def recv(self) -> Union[str, bytes]: ...
    async def close(self) -> None: ...


Connector = Callable[[str, dict[str, str]], Awaitable[WebSocketLike]]
EventHandler = Callable[[SttEvent, float], None]


async def _websockets_connect(url: str, headers: dict[str, str]) -> WebSocketLike:
    from websockets.asyncio.client import connect  # imported lazily: only needed for real calls

    return await connect(url, additional_headers=headers, open_timeout=CONNECT_TIMEOUT_S)


def _describe(exc: BaseException) -> str:
    """Short, credential-free description of a connection failure."""
    text = str(exc).splitlines()[0][:200] if str(exc) else ""
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


class DeepgramStream:
    """One live transcription stream (= one call)."""

    def __init__(
        self,
        api_key: str,
        params: DeepgramParams,
        on_event: EventHandler,
        clock: Optional[AudioClock] = None,
        connect: Connector = _websockets_connect,
        keepalive_after_idle_s: float = KEEPALIVE_AFTER_IDLE_S,
    ) -> None:
        self._api_key = api_key
        self._params = params
        self._on_event = on_event
        self._connect = connect
        self._keepalive_s = keepalive_after_idle_s
        self.clock = clock if clock is not None else AudioClock()
        self._queue: asyncio.Queue[Optional[tuple[bytes, float]]] = asyncio.Queue()
        self._task: Optional[asyncio.Task[None]] = None
        self.failed = False
        self.dropped_frames = 0
        self.keepalives_sent = 0

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    def send_audio(self, payload: bytes, arrival_wall: Optional[float] = None) -> None:
        """Queue one audio frame. Never blocks, never raises."""
        if self.failed or self._task is None:
            return
        if self._queue.qsize() >= MAX_QUEUED_FRAMES:
            self.dropped_frames += 1
            return
        self._queue.put_nowait((payload, time.time() if arrival_wall is None else arrival_wall))

    async def close(self) -> None:
        """Flush queued audio, send CloseStream, wait briefly for the last results."""
        if self._task is None:
            return
        self._queue.put_nowait(None)
        try:
            await asyncio.wait_for(asyncio.shield(self._task), CLOSE_GRACE_S + CONNECT_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._task.cancel()
        except Exception:  # _run already logged it
            pass

    # -- internals
    def _fail(self, stage: str, exc: BaseException) -> None:
        if self.failed:
            return
        self.failed = True
        message = f"Deepgram {stage} failed: {_describe(exc)}"
        log.error("%s - transcription is off for the rest of this call", message)
        self._on_event(SttError(message), time.time())

    async def _run(self) -> None:
        try:
            ws = await self._connect(self._params.url(), {"Authorization": f"Token {self._api_key}"})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail("connect", exc)
            return
        loop = asyncio.get_running_loop()
        sender = loop.create_task(self._send_loop(ws))
        receiver = loop.create_task(self._receive(ws))
        try:
            done, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
            if sender in done and sender.exception() is None:
                # CloseStream sent: Deepgram flushes its last results, then closes.
                await asyncio.wait({receiver}, timeout=CLOSE_GRACE_S)
            else:
                # Send failed, or Deepgram went away before we asked it to stop.
                first = sender if sender in done else receiver
                self._fail("connection", first.exception() or ConnectionError("closed by Deepgram"))
        finally:
            sender.cancel()
            receiver.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)
            try:
                await ws.close()
            except Exception:
                pass

    async def _send_loop(self, ws: WebSocketLike) -> None:
        while True:
            try:
                item = await asyncio.wait_for(self._queue.get(), self._keepalive_s)
            except asyncio.TimeoutError:
                await ws.send(json.dumps({"type": "KeepAlive"}))
                self.keepalives_sent += 1
                continue
            if item is None:
                await ws.send(json.dumps({"type": "CloseStream"}))
                return
            payload, arrival_wall = item
            await ws.send(payload)
            self.clock.mark(audio_duration_s(payload), arrival_wall)

    async def _receive(self, ws: WebSocketLike) -> None:
        """Read until the connection closes (recv then raises; _run decides if that is bad)."""
        while True:
            raw = await ws.recv()
            self._on_event(parse_message(raw), time.time())
