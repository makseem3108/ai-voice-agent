"""Deepgram Flux: conversational speech-to-text with built-in turn detection.

Unlike the phase 2a model (nova-3), Flux does not just transcribe: it tells us
*when the caller's turn starts and ends* (``/v2/listen``). The events are:

* ``StartOfTurn``     - the caller began speaking (used for barge-in)
* ``EagerEndOfTurn``  - they have *probably* finished (we may start thinking)
* ``TurnResumed``     - no, they kept talking (discard that head start)
* ``EndOfTurn``       - they finished; ``transcript`` is the whole turn

Message parsing is pure; :class:`FluxStream` does the WebSocket I/O and, like
the phase 2a client, never raises into the call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Union
from urllib.parse import urlencode

from .audiosocket import SAMPLE_RATE_HZ, audio_duration_s
from .stt_deepgram import (
    CLOSE_GRACE_S,
    CONNECT_TIMEOUT_S,
    MAX_QUEUED_FRAMES,
    AudioClock,
    Connector,
    EventHandler,
    SttError,
    WebSocketLike,
    _describe,
    _websockets_connect,
)

log = logging.getLogger("voice_agent.stt")

FLUX_URL = "wss://api.deepgram.com/v2/listen"
#: Deepgram recommends ~80 ms chunks for Flux: 4 AudioSocket frames of 20 ms.
CHUNK_BYTES = 1280

START_OF_TURN = "StartOfTurn"
UPDATE = "Update"
EAGER_END_OF_TURN = "EagerEndOfTurn"
TURN_RESUMED = "TurnResumed"
END_OF_TURN = "EndOfTurn"


@dataclass(frozen=True)
class FluxParams:
    model: str = "flux-general-en"
    eot_threshold: float = 0.7
    #: 0 disables eager end-of-turn (no EagerEndOfTurn / TurnResumed events).
    eager_eot_threshold: float = 0.5
    eot_timeout_ms: int = 3000
    #: Domain words/phrases Flux should listen for ("keyterm prompting").
    keyterms: tuple[str, ...] = ()

    def query(self) -> list[tuple[str, str]]:
        params = [
            ("model", self.model),
            ("encoding", "linear16"),
            ("sample_rate", str(SAMPLE_RATE_HZ)),
            ("eot_threshold", f"{self.eot_threshold:g}"),
            ("eot_timeout_ms", str(self.eot_timeout_ms)),
        ]
        if self.eager_eot_threshold > 0:
            params.append(("eager_eot_threshold", f"{min(self.eager_eot_threshold, self.eot_threshold):g}"))
        # Flux takes one `keyterm` parameter per term; multi-word phrases are URL-encoded.
        params += [("keyterm", term) for term in self.keyterms]
        return params

    def url(self) -> str:
        """Request URL. Contains no credentials, so it is safe to log."""
        return f"{FLUX_URL}?{urlencode(self.query())}"


@dataclass(frozen=True)
class TurnEvent:
    """A Flux ``TurnInfo`` message."""

    event: str
    turn_index: int
    transcript: str
    eot_confidence: float = 0.0
    audio_window_end: float = 0.0  # audio seconds
    audio_window_start: float = 0.0
    last_word_end: Optional[float] = None  # audio seconds

    @property
    def speech_end(self) -> float:
        """Best estimate (audio seconds) of when the caller's last word ended."""
        return self.last_word_end if self.last_word_end is not None else self.audio_window_end


@dataclass(frozen=True)
class FluxInfo:
    """Connected / Warning / other non-turn messages."""

    type: str
    detail: str = ""


FluxEvent = Union[TurnEvent, FluxInfo, SttError]


def _f(value: Any, default: float = 0.0) -> float:
    """Flux sends some floats as strings."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_flux_message(raw: Union[str, bytes]) -> FluxEvent:
    """Turn one Flux WebSocket message into an event. Never raises."""
    try:
        msg = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return SttError("unparseable message from Deepgram Flux")
    if not isinstance(msg, dict):
        return SttError("unexpected message shape from Deepgram Flux")
    kind = msg.get("type")
    if kind == "TurnInfo":
        words = msg.get("words") or []
        last_end: Optional[float] = None
        if isinstance(words, list) and words and isinstance(words[-1], dict) and "end" in words[-1]:
            last_end = _f(words[-1]["end"])
        return TurnEvent(
            event=str(msg.get("event") or ""),
            turn_index=int(_f(msg.get("turn_index"))),
            transcript=str(msg.get("transcript") or "").strip(),
            eot_confidence=_f(msg.get("end_of_turn_confidence")),
            audio_window_end=_f(msg.get("audio_window_end")),
            audio_window_start=_f(msg.get("audio_window_start")),
            last_word_end=last_end,
        )
    if kind == "Error":
        return SttError(f"{msg.get('code') or 'ERROR'}: {msg.get('description') or 'no description'}")
    if kind in ("Warning", "ConfigureFailure"):
        return FluxInfo(str(kind), f"{msg.get('code') or ''} {msg.get('description') or ''}".strip())
    return FluxInfo(str(kind))


class FluxStream:
    """One Flux stream (= one call): 20 ms frames in, turn events out."""

    def __init__(
        self,
        api_key: str,
        params: FluxParams,
        on_event: EventHandler,
        clock: Optional[AudioClock] = None,
        connect: Connector = _websockets_connect,
    ) -> None:
        self._api_key = api_key
        self._params = params
        self._on_event = on_event
        self._connect = connect
        self.clock = clock if clock is not None else AudioClock()
        self._queue: asyncio.Queue[Optional[tuple[bytes, float]]] = asyncio.Queue()
        self._task: Optional[asyncio.Task[None]] = None
        self.failed = False
        self.dropped_frames = 0

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
        if self._task is None:
            return
        self._queue.put_nowait(None)
        try:
            await asyncio.wait_for(asyncio.shield(self._task), CLOSE_GRACE_S + CONNECT_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._task.cancel()
        except Exception:
            pass

    def _fail(self, stage: str, exc: BaseException) -> None:
        if self.failed:
            return
        self.failed = True
        message = f"Deepgram Flux {stage} failed: {_describe(exc)}"
        log.error("%s - the agent can no longer hear the caller", message)
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
                await asyncio.wait({receiver}, timeout=1.0)  # CloseStream sent; brief drain
            else:
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
        chunk = bytearray()
        marks: list[tuple[float, float]] = []  # (frame seconds, arrival wall) inside the chunk

        async def flush() -> None:
            if chunk:
                await ws.send(bytes(chunk))
                for seconds, wall in marks:
                    self.clock.mark(seconds, wall)
                chunk.clear()
                marks.clear()

        while True:
            item = await self._queue.get()
            if item is None:
                await flush()
                await ws.send(json.dumps({"type": "CloseStream"}))
                return
            payload, arrival_wall = item
            chunk += payload
            marks.append((audio_duration_s(payload), arrival_wall))
            if len(chunk) >= CHUNK_BYTES:
                await flush()

    async def _receive(self, ws: WebSocketLike) -> None:
        while True:
            raw = await ws.recv()
            self._on_event(parse_flux_message(raw), time.time())
