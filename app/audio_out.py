"""Outbound audio pacer: one 20 ms frame to the caller every 20 ms.

Speech synthesis delivers audio much faster than real time, in chunks of any
size. Asterisk plays whatever we send as soon as it arrives and has no jitter
buffer on this path, so *we* must release audio at exactly the speed it should
be heard: one 320-byte frame every 20 ms.

Timing uses absolute deadlines on a monotonic clock: frame *n* of a burst is
due at ``start + n * 20 ms``. Sleeping "20 ms after the previous frame" would
let every small delay add up (drift); deadlines measured from the start of the
burst cannot drift, because a late frame simply shortens the next sleep.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Optional

FRAME_BYTES = 320  # 160 samples of 16-bit audio = 20 ms at 8 kHz
FRAME_SECONDS = 0.020
#: If we fall further behind than this (event loop stall), restart the schedule
#: instead of sending a burst of frames to catch up.
MAX_LAG_SECONDS = 0.100
#: After a pause that turned out to be a false alarm: with less than this left of
#: the sentence, the rest is dropped; with more, the sentence starts again.
SHORT_REST_SECONDS = 1.0
BYTES_PER_SECOND = 16000  # 8 kHz, 16-bit

SendFrame = Callable[[bytes], None]


class Pacer:
    def __init__(
        self,
        send_frame: SendFrame,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        on_frame: Optional[Callable[[], None]] = None,
    ) -> None:
        self._send_frame = send_frame
        self._clock = clock
        self._sleep = sleep
        self._on_frame = on_frame
        self._buf = bytearray()
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._task: Optional[asyncio.Task[None]] = None
        self._in_burst = False
        self._paused = False
        self.frames_sent = 0
        self.resyncs = 0
        # Sentence bookkeeping, for "never continue in the middle of a word":
        # the audio of the current sentence that was already sent, how much of it
        # is still queued, and the lengths of the sentences queued behind it.
        self._played = bytearray()
        self._left = 0
        self._later: list[int] = []

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self._buf.clear()
        self._played.clear()
        self._left = 0
        self._later.clear()
        self._idle.set()

    def push(self, pcm: bytes, *, new_sentence: bool = False) -> None:
        """Queue audio of any length (chunks need not align to frames).

        ``new_sentence``: this chunk starts a new sentence (one synthesis request
        or one fixed phrase) - the unit that is restarted after a false pause.
        """
        if not pcm:
            return
        if new_sentence and (self._left > 0 or self._later):
            self._later.append(len(pcm))
        elif self._later:
            self._later[-1] += len(pcm)
        else:
            if new_sentence:
                self._played.clear()
                self._left = 0
            self._left += len(pcm)
        self._buf += pcm
        self._idle.clear()
        self._wake.set()

    def flush(self) -> None:
        """End of an utterance: pad the last partial frame with silence so it is sent."""
        remainder = len(self._buf) % FRAME_BYTES
        if remainder:
            pad = FRAME_BYTES - remainder
            self._buf += bytes(pad)
            if self._later:
                self._later[-1] += pad
            else:
                self._left += pad
            self._wake.set()

    def pause(self) -> None:
        """Stop releasing audio but keep the queue (the caller may be interrupting)."""
        self._paused = True

    def resume(self) -> None:
        """Continue after :meth:`pause`, exactly where the audio stopped."""
        if self._paused:
            self._paused = False
            self._wake.set()

    def resume_cleanly(self) -> str:
        """Continue after a pause that was a false alarm, without starting mid-word.

        Less than a second of the sentence left: that rest is dropped ("dropped").
        More: the sentence is said again from its beginning ("restarted").
        Sentences queued behind it follow as usual.
        """
        if not self._paused:
            return "running"
        left = min(self._left, len(self._buf))
        if left < SHORT_REST_SECONDS * BYTES_PER_SECOND:
            del self._buf[:left]
            self._left = 0
            self._played.clear()
            self._next_sentence()
            action = "dropped"
        else:
            self._buf[:0] = self._played
            self._left = left + len(self._played)
            self._played.clear()
            action = "restarted"
        remainder = len(self._buf) % FRAME_BYTES
        if remainder and not self._later:
            self._buf += bytes(FRAME_BYTES - remainder)
            self._left += FRAME_BYTES - remainder
        if not self._buf and not self._in_burst:
            self._idle.set()
        self.resume()
        return action

    def _next_sentence(self) -> None:
        while self._left <= 0 and self._later:
            self._left += self._later.pop(0)
            self._played.clear()

    @property
    def paused(self) -> bool:
        return self._paused

    def clear(self) -> int:
        """Barge-in: drop everything not yet sent. Returns the number of frames dropped."""
        dropped = len(self._buf) // FRAME_BYTES
        self._buf.clear()
        self._played.clear()
        self._left = 0
        self._later.clear()
        if not self._in_burst:
            self._idle.set()
        return dropped

    @property
    def speaking(self) -> bool:
        """True while audio is queued or being released to the caller."""
        return self._in_burst or len(self._buf) >= FRAME_BYTES

    @property
    def queued_seconds(self) -> float:
        return (len(self._buf) // FRAME_BYTES) * FRAME_SECONDS

    async def drained(self) -> None:
        """Wait until everything queued has been sent (or cleared)."""
        await self._idle.wait()

    # ------------------------------------------------------------------ loop
    async def _run(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            if self._paused or len(self._buf) < FRAME_BYTES:
                continue
            start = self._clock()
            sent = 0
            self._in_burst = True
            try:
                while len(self._buf) >= FRAME_BYTES and not self._paused:
                    frame = bytes(self._buf[:FRAME_BYTES])
                    del self._buf[:FRAME_BYTES]
                    self._played += frame
                    self._left -= FRAME_BYTES
                    if self._left <= 0:
                        self._next_sentence()
                    self._send_frame(frame)
                    self.frames_sent += 1
                    sent += 1
                    if self._on_frame is not None:
                        self._on_frame()
                    delay = start + sent * FRAME_SECONDS - self._clock()
                    if delay > 0:
                        await self._sleep(delay)
                    elif delay < -MAX_LAG_SECONDS:
                        start = self._clock() - sent * FRAME_SECONDS
                        self.resyncs += 1
            finally:
                self._in_burst = False
            if not self._buf:
                self._idle.set()
