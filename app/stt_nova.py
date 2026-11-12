"""Deepgram Nova-3 as a turn-taking speech-to-text provider (Hebrew, Arabic, ...).

Flux, the English model, tells us when the caller's turn starts and ends.
Nova-3 does not: it only sends pieces of transcript. This module turns those
pieces into the same turn events the agent already understands, so nothing
else in the program has to know which model is listening.

What Nova-3 sends (``/v1/listen``, see :mod:`app.stt_deepgram` for the I/O):

* **interim results** - a guess for the audio so far; it may still change;
* **final results** (``is_final``) - that slice of audio will not change any
  more. A long sentence arrives as several final slices;
* ``speech_final`` on a result - the caller paused for ``endpointing`` ms
  (300 by default): the normal "they finished" signal;
* ``UtteranceEnd`` - no words for ``utterance_end_ms`` (1 s). A slower backup
  that still works when background noise keeps the pause detector busy.

How that becomes turns (:class:`NovaTurnTracker`, plain code, no I/O):

* first words of a new utterance            -> ``StartOfTurn`` (confirms barge-in)
* every interim / final slice               -> ``Update`` (text so far)
* final slices are **joined** until ``speech_final`` -> ``EndOfTurn`` with the whole text
* ``UtteranceEnd`` while a turn is open     -> ``EndOfTurn`` (backup)
* nothing new for ``turn_timeout`` seconds  -> ``EndOfTurn`` (our own last resort)

There is no speculative "eager" end of turn here, so replies start a little
later than with Flux; the ``[turn]`` log lines show by how much.

Each ``EndOfTurn`` carries ``details``: what ended the turn, and for every
slice the last interim guess next to the final text, with the words that were
in the guess but not in the final turn (``lost_words``). That makes it visible
when the model drops a word between guessing and finalizing.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Optional

from .speech import (
    END_OF_TURN,
    START_OF_TURN,
    UPDATE,
    SpeechEvent,
    SpeechEventHandler,
    SttError,
    TurnEvent,
)
from .stt_deepgram import (
    AudioClock,
    Connector,
    DeepgramParams,
    DeepgramStream,
    SttEvent,
    Transcript,
    UtteranceEnd,
    _websockets_connect,
)

log = logging.getLogger("voice_agent.stt")

ENDED_BY_SPEECH_FINAL = "speech_final"
ENDED_BY_UTTERANCE_END = "utterance_end"
ENDED_BY_TIMEOUT = "timeout"
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _words(text: str) -> list[str]:
    return [w.casefold() for w in _WORD_RE.findall(text)]


def _is_lost(word: str, final_words: list[str]) -> bool:
    """Was this word of an interim guess really dropped from the final text?

    A guess often shows a word half-heard ("اساعد" before "اساعدك") or with a
    joined particle the final text writes differently ("והביקור" / "הביקור").
    Those are the same word, not a loss, so a word only counts as lost if no
    final word contains it or is contained in it.
    """
    return not any(word in final or final in word for final in final_words if len(final) > 1 or final == word)


class NovaTurnTracker:
    """Turns Nova-3 results into turn events. Pure: feed events in, get events out."""

    def __init__(self) -> None:
        self.turn_index = 0
        self._open = False
        self._finals: list[str] = []
        self._segments: list[dict[str, Any]] = []
        self._interim = ""  # latest guess for the slice not yet finalized
        self._window_start = 0.0
        self._window_end = 0.0
        self._last_word_end: Optional[float] = None

    @property
    def open(self) -> bool:
        """A turn has started and not ended yet."""
        return self._open

    def _text(self, extra: str = "") -> str:
        return " ".join(part for part in (*self._finals, extra) if part)

    def _event(self, kind: str, text: str, confidence: float = 0.0, details: Optional[dict[str, Any]] = None) -> TurnEvent:
        return TurnEvent(
            event=kind,
            turn_index=self.turn_index,
            transcript=text,
            eot_confidence=confidence,
            audio_window_end=self._window_end,
            audio_window_start=self._window_start,
            last_word_end=self._last_word_end,
            details=details,
        )

    def feed(self, event: SttEvent) -> list[TurnEvent]:
        """One message from Deepgram -> zero or more turn events."""
        if isinstance(event, UtteranceEnd):
            # The signal names the last word it is about. If that word is older than
            # the turn now open (it belongs to speech we already closed), or Deepgram
            # marks it as already finalized (-1), it must not end the new turn.
            if event.last_word_end < 0 or event.last_word_end <= self._window_start:
                return []
            return self._end(ENDED_BY_UTTERANCE_END)
        if not isinstance(event, Transcript):
            return []
        out: list[TurnEvent] = []
        text = event.text.strip()
        if text:
            if not self._open:
                self._open = True
                self._window_start = event.start
                self._window_end = event.end
                out.append(self._event(START_OF_TURN, text))
            self._window_end = max(self._window_end, event.end)
            if event.is_final:
                self._segments.append({"interim": self._interim or None, "final": text})
                self._finals.append(text)
                self._interim = ""
                self._last_word_end = event.last_word_end if event.last_word_end is not None else event.end
                out.append(self._event(UPDATE, self._text()))
            else:
                self._interim = text
                out.append(self._event(UPDATE, self._text(text)))
        if event.is_final and event.speech_final:
            out += self._end(ENDED_BY_SPEECH_FINAL)
        return out

    def timeout(self) -> list[TurnEvent]:
        """Our own rule: nothing new arrived for a while although a turn is open."""
        return self._end(ENDED_BY_TIMEOUT)

    def _end(self, ended_by: str) -> list[TurnEvent]:
        if not self._open:
            return []
        if self._interim:
            # The last slice never became final; better the guess than losing the words.
            self._segments.append({"interim": self._interim, "final": None})
            self._finals.append(self._interim)
            self._interim = ""
        text = self._text()
        final_words = _words(text)
        lost: list[str] = []
        for segment in self._segments:
            if segment["final"] is not None and segment["interim"]:
                lost += [w for w in _words(segment["interim"]) if _is_lost(w, final_words) and w not in lost]
        details = {"ended_by": ended_by, "segments": self._segments, "lost_words": lost}
        confidence = 1.0 if ended_by == ENDED_BY_SPEECH_FINAL else 0.5
        events = [self._event(END_OF_TURN, text, confidence, details)] if text else []
        self.turn_index += 1
        self._open = False
        self._finals, self._segments = [], []
        self._last_word_end = None
        return events


class NovaStream:
    """One Nova-3 stream (= one call). Implements :class:`~app.speech.STTProvider`."""

    name = "nova3"
    #: Turns end on a plain pause, so the agent should join a turn with what the
    #: caller says right after it, as long as the reply has not started.
    merges_turns = True

    def __init__(
        self,
        api_key: str,
        params: DeepgramParams,
        on_event: SpeechEventHandler,
        clock: Optional[AudioClock] = None,
        connect: Connector = _websockets_connect,
        turn_timeout_s: float = 2.5,
    ) -> None:
        self._params = params
        self._on_event = on_event
        self._turn_timeout_s = turn_timeout_s
        self._tracker = NovaTurnTracker()
        self._timer: Optional[asyncio.TimerHandle] = None
        self._inner = DeepgramStream(api_key, params, self._on_deepgram, clock, connect)
        self.clock = self._inner.clock
        self.turns_by: dict[str, int] = {ENDED_BY_SPEECH_FINAL: 0, ENDED_BY_UTTERANCE_END: 0, ENDED_BY_TIMEOUT: 0}

    @property
    def model(self) -> str:
        return f"{self._params.model}/{self._params.language}"

    @property
    def failed(self) -> bool:
        return self._inner.failed

    @property
    def supports_eager(self) -> bool:
        return False  # no speculative head starts with Nova-3 (for now)

    def start(self) -> None:
        self._inner.start()

    def send_audio(self, payload: bytes, arrival_wall: Optional[float] = None) -> None:
        self._inner.send_audio(payload, arrival_wall)

    async def close(self) -> None:
        self._cancel_timer()
        await self._inner.close()

    # ------------------------------------------------------------------ events
    def _on_deepgram(self, event: SttEvent, arrival_wall: float) -> None:
        if isinstance(event, SttError):
            self._cancel_timer()
            self._on_event(event, arrival_wall)
            return
        self._emit(self._tracker.feed(event), arrival_wall)
        if isinstance(event, Transcript) and event.text.strip():
            self._arm_timer()  # words keep coming: push our own deadline out

    def _emit(self, events: list[TurnEvent], arrival_wall: float) -> None:
        for turn_event in events:
            if turn_event.event == END_OF_TURN:
                self._cancel_timer()
                ended_by = (turn_event.details or {}).get("ended_by", "")
                self.turns_by[ended_by] = self.turns_by.get(ended_by, 0) + 1
            forward: SpeechEvent = turn_event
            self._on_event(forward, arrival_wall)

    def _arm_timer(self) -> None:
        self._cancel_timer()
        if self._tracker.open and self._turn_timeout_s > 0:
            self._timer = asyncio.get_running_loop().call_later(self._turn_timeout_s, self._on_timeout)

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _on_timeout(self) -> None:
        self._timer = None
        try:
            events = self._tracker.timeout()
            if events:
                log.warning("Nova-3 sent no end-of-speech signal for %.1f s; ending the turn ourselves", self._turn_timeout_s)
            self._emit(events, time.time())
        except Exception:
            log.exception("turn timeout handling failed")
