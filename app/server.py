"""AudioSocket TCP server (asyncio). Current behaviour: delayed echo.

Asterisk opens one TCP connection per call (dialplan ``AudioSocket()``), sends
the call UUID, then streams caller audio. We send every audio frame back after
a fixed delay, so the caller hears themselves about one second late.

Run with ``python -m app.server`` from the project root.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .audiosocket import (
    Frame,
    FrameParser,
    FrameType,
    audio_duration_s,
    encode_frame,
    parse_uuid,
)
from .config import Settings, load_settings

log = logging.getLogger("voice_agent")

READ_CHUNK = 4096


@dataclass
class CallStats:
    """Per-call counters, reported in one log line when the call ends."""

    started: float = field(default_factory=time.monotonic)
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
            s = self.stats
            log.info(
                "call ended uuid=%s duration=%.2fs frames_in=%d frames_out=%d "
                "avg_jitter_ms=%.2f reason=%s",
                s.call_uuid or "unknown",
                s.duration_s(),
                s.frames_in,
                s.frames_out,
                s.avg_jitter_ms,
                reason,
            )

    async def _read_loop(self) -> str:
        while True:
            data = await self._reader.read(READ_CHUNK)
            if not data:
                return "eof"
            now = time.monotonic()
            for frame in self._parser.feed(data):
                done = self._handle(frame, now)
                if done is not None:
                    return done

    def _handle(self, frame: Frame, now: float) -> Optional[str]:
        """Process one frame. Returns an end-of-call reason, or None to go on."""
        kind = frame.kind
        if kind is FrameType.AUDIO:
            self.stats.audio_in(frame.payload, now)
            self._echo_later(frame.payload)
            return None
        if kind is FrameType.UUID:
            try:
                self.stats.call_uuid = parse_uuid(frame.payload)
            except ValueError as exc:
                log.warning("bad UUID frame: %s", exc)
                return "bad-uuid"
            log.info("call started uuid=%s", self.stats.call_uuid)
            return None
        if kind is FrameType.HANGUP:
            return "hangup"
        if kind is FrameType.ERROR:
            code = frame.payload[0] if frame.payload else None
            return f"asterisk-error:{code}"
        self.stats.unknown_frames += 1
        log.debug("ignoring unknown frame type 0x%02x (%d bytes)", frame.type, len(frame.payload))
        return None

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
    log.info("AudioSocket server listening on %s (echo delay %d ms)", bound, settings.echo_delay_ms)

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
    asyncio.run(serve(settings))


if __name__ == "__main__":
    main()
