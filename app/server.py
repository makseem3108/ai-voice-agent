"""AudioSocket TCP server (asyncio).

Asterisk opens one TCP connection per call (dialplan ``AudioSocket()``), sends
the call UUID, then streams caller audio in 20 ms frames. Each frame is
forwarded to Deepgram for live transcription; transcripts are logged and saved
per call. Nothing is sent back, so the caller hears silence - unless
``ECHO_MODE=true``, which plays their own audio back after a delay (debugging).

Run with ``python -m app.server`` from the project root.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .audiosocket import (
    Frame,
    FrameParser,
    FrameType,
    audio_duration_s,
    encode_frame,
    parse_uuid,
)
from .call_log import CallRecorder, format_record
from .config import Settings, load_settings
from .stt_deepgram import AudioClock, DeepgramParams, DeepgramStream, SttEvent, TranscriptTracker

log = logging.getLogger("voice_agent")

READ_CHUNK = 4096


@dataclass
class CallStats:
    """Per-call counters, reported in one log line when the call ends."""

    started: float = field(default_factory=time.monotonic)
    started_wall: float = field(default_factory=time.time)
    call_uuid: Optional[uuid.UUID] = None
    frames_in: int = 0
    frames_out: int = 0
    unknown_frames: int = 0
    _last_arrival: Optional[float] = None
    _expected_gap_s: float = 0.0
    _jitter_sum_s: float = 0.0
    _jitter_samples: int = 0

    def audio_in(self, payload: bytes, now: float) -> None:
        """Count an inbound audio frame and update the jitter estimate.

        Jitter here = how far each inter-arrival gap is from the gap the audio
        itself implies (20 ms of audio should arrive 20 ms after the previous
        frame). We report the mean of the absolute deviations.
        """
        self.frames_in += 1
        if self._last_arrival is not None:
            gap = now - self._last_arrival
            self._jitter_sum_s += abs(gap - self._expected_gap_s)
            self._jitter_samples += 1
        self._last_arrival = now
        self._expected_gap_s = audio_duration_s(payload)

    @property
    def avg_jitter_ms(self) -> float:
        if not self._jitter_samples:
            return 0.0
        return 1000.0 * self._jitter_sum_s / self._jitter_samples

    def duration_s(self, now: Optional[float] = None) -> float:
        return (time.monotonic() if now is None else now) - self.started


def _ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0f}"


class CallSession:
    """One AudioSocket connection = one call."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        settings: Settings,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._settings = settings
        self._parser = FrameParser()
        self._loop = asyncio.get_running_loop()
        self._pending: set[asyncio.TimerHandle] = set()
        self.stats = CallStats()
        self._tag = "--------"
        self._recorder: Optional[CallRecorder] = None
        self._clock = AudioClock()
        self._tracker = TranscriptTracker(self._clock)
        self._stt: Optional[DeepgramStream] = None

    async def run(self) -> None:
        reason = "eof"
        try:
            reason = await self._read_loop()
        except (ConnectionError, asyncio.IncompleteReadError) as exc:
            reason = f"connection-error:{type(exc).__name__}"
        except asyncio.CancelledError:
            reason = "server-shutdown"
            raise
        finally:
            self._cancel_pending()
            await self._close()
            if self._stt is not None:
                try:
                    await self._stt.close()  # flush: last finals arrive here
                except asyncio.CancelledError:
                    pass
            self._finish(reason)

    # ------------------------------------------------------------------ call start / end
    def _start_call(self, call_uuid: uuid.UUID) -> None:
        self.stats.call_uuid = call_uuid
        self._tag = str(call_uuid)[:8]
        s = self._settings
        self._recorder = CallRecorder(s.calls_dir, str(call_uuid), self.stats.started_wall)
        mode = "stt" if s.stt_active else "no-stt"
        if s.echo_mode:
            mode += "+echo"
        log.info("call started uuid=%s mode=%s", call_uuid, mode)
        self._record({"type": "call_start", "uuid": str(call_uuid), "mode": mode})
        if s.stt_active:
            params = DeepgramParams(
                model=s.deepgram_model,
                language=s.deepgram_language,
                endpointing_ms=s.deepgram_endpointing_ms,
                utterance_end_ms=s.deepgram_utterance_end_ms,
            )
            self._stt = DeepgramStream(s.deepgram_api_key, params, self._on_stt_event, self._clock)
            self._stt.start()
        elif s.stt_enabled:
            log.warning("[%s] DEEPGRAM_API_KEY is not set - transcription is off", self._tag)

    def _finish(self, reason: str) -> None:
        s, t = self.stats, self._tracker
        summary: dict[str, Any] = {
            "type": "call_end",
            "reason": reason,
            "duration_s": round(s.duration_s(), 2),
            "frames_in": s.frames_in,
            "frames_out": s.frames_out,
            "avg_jitter_ms": round(s.avg_jitter_ms, 2),
            "finals": t.finals,
            "stt_latency_avg_ms": None if t.avg_latency_ms is None else round(t.avg_latency_ms, 1),
            "stt_latency_max_ms": None if t.max_latency_ms is None else round(t.max_latency_ms, 1),
            "audio_sent_s": round(self._clock.total_seconds, 2),
            "deepgram_reported_s": t.deepgram_duration,
            "deepgram_request_id": t.request_id or None,
            "stt_failed": bool(self._stt and self._stt.failed),
            "stt_dropped_frames": self._stt.dropped_frames if self._stt else 0,
        }
        self._record(summary)
        if self._recorder is not None:
            self._recorder.close()
        log.info(
            "call ended uuid=%s duration=%.2fs finals=%d stt_latency_avg_ms=%s stt_latency_max_ms=%s "
            "audio_sent_s=%.2f frames_in=%d frames_out=%d avg_jitter_ms=%.2f reason=%s",
            s.call_uuid or "unknown",
            s.duration_s(),
            t.finals,
            _ms(t.avg_latency_ms),
            _ms(t.max_latency_ms),
            self._clock.total_seconds,
            s.frames_in,
            s.frames_out,
            s.avg_jitter_ms,
            reason,
        )

    # ------------------------------------------------------------------ inbound frames
    async def _read_loop(self) -> str:
        while True:
            data = await self._reader.read(READ_CHUNK)
            if not data:
                return "eof"
            now = time.monotonic()
            wall = time.time()
            for frame in self._parser.feed(data):
                done = self._handle(frame, now, wall)
                if done is not None:
                    return done

    def _handle(self, frame: Frame, now: float, wall: float) -> Optional[str]:
        """Process one frame. Returns an end-of-call reason, or None to go on."""
        kind = frame.kind
        if kind is FrameType.AUDIO:
            self.stats.audio_in(frame.payload, now)
            if self._stt is not None:
                self._stt.send_audio(frame.payload, wall)
            if self._settings.echo_mode:
                self._echo_later(frame.payload)
            return None
        if kind is FrameType.UUID:
            try:
                self._start_call(parse_uuid(frame.payload))
            except ValueError as exc:
                log.warning("bad UUID frame: %s", exc)
                return "bad-uuid"
            return None
        if kind is FrameType.HANGUP:
            return "hangup"
        if kind is FrameType.ERROR:
            code = frame.payload[0] if frame.payload else None
            return f"asterisk-error:{code}"
        self.stats.unknown_frames += 1
        log.debug("ignoring unknown frame type 0x%02x (%d bytes)", frame.type, len(frame.payload))
        return None

    # ------------------------------------------------------------------ transcription
    def _on_stt_event(self, event: SttEvent, arrival_wall: float) -> None:
        """Called by the Deepgram stream for every message. Must never raise."""
        try:
            for record in self._tracker.on_event(event, arrival_wall):
                self._record(record, arrival_wall)
                text = format_record(record)
                if text is not None:
                    log.info("[%s] %s", self._tag, text)
        except Exception:
            log.exception("[%s] failed to handle STT event", self._tag)

    def _record(self, record: dict[str, Any], wall: Optional[float] = None) -> None:
        if self._recorder is not None:
            self._recorder.write(record, wall)

    # ------------------------------------------------------------------ debug echo
    def _echo_later(self, payload: bytes) -> None:
        handle: Optional[asyncio.TimerHandle] = None

        def send() -> None:
            if handle is not None:
                self._pending.discard(handle)
            if self._writer.is_closing():
                return
            self._writer.write(encode_frame(FrameType.AUDIO, payload))
            self.stats.frames_out += 1

        # call_later fires in scheduling order for equal delays, so frame order is kept.
        handle = self._loop.call_later(self._settings.echo_delay_s, send)
        self._pending.add(handle)

    def _cancel_pending(self) -> None:
        for handle in self._pending:
            handle.cancel()
        self._pending.clear()

    async def _close(self) -> None:
        self._writer.close()
        try:
            await self._writer.wait_closed()
        except (ConnectionError, OSError):
            pass


async def serve(settings: Settings) -> None:
    """Run the server until SIGTERM / SIGINT."""

    async def on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await CallSession(reader, writer, settings).run()

    server = await asyncio.start_server(on_connect, host=settings.host, port=settings.port)
    bound = ", ".join(f"{s.getsockname()[0]}:{s.getsockname()[1]}" for s in server.sockets or ())
    log.info(
        "AudioSocket server listening on %s (stt=%s model=%s echo_mode=%s)",
        bound,
        "on" if settings.stt_active else "off",
        settings.deepgram_model,
        settings.echo_mode,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    async with server:
        await stop.wait()
        log.info("shutting down")


def main() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # At DEBUG the websockets library logs handshake headers, which would
    # include the Authorization header. Never let it go below INFO.
    logging.getLogger("websockets").setLevel(logging.INFO)
    asyncio.run(serve(settings))


if __name__ == "__main__":
    main()
