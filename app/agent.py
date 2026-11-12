"""Conversation engine: one :class:`Agent` per call.

Who decides what
----------------
The **code** owns the flow of the call; the **language model** only extracts
facts and phrases sentences.

Every model reply is structured: ``{"slots": {...}, "say": "..."}``. The slots
are validated and merged into the call's state (:mod:`app.slots`), and from
that state the code decides:

* the **outcome** - a disqualifying slot value means *not a fit*, all required
  slots filled means *qualified*;
* the **next question** - the first slot still missing; the model is told which
  one to ask and only words it naturally;
* **"anything else?"** - asked by the server, exactly once, right after the
  outcome has been announced;
* the **closing line** - a fixed phrase chosen by the code from the outcome,
  played only after the caller's own words were a decline or a goodbye. If the
  caller says goodbye before the outcome was announced, their last sentence is
  first read for slot values ("Yes. Thank you. Goodbye." answers the pending
  question too), and only then is the outcome computed and the closing chosen.

Only ``say`` is ever sent to text-to-speech.

Turn flow::

    caller stops talking -> EndOfTurn (transcript)
        -> plan from the current slots (what should this reply do?)
        -> LLM returns {slots, say}
        -> merge slots, plan again; if the reply no longer fits, ask once more
        -> say is split into sentences -> TTS -> pacer (20 ms frames)

Head start ("eager end of turn"), two-stage barge-in and the cancellable
hang-up work as before; see docs/phase2b-agent.md.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Callable, Optional, Protocol

from .audio_out import Pacer
from .intent import DECLINE, GOODBYE, caller_reply_kind, interrupts_closing
from .llm_groq import PATH_FUNCTION, PATH_JSON, PATH_TEXT, LLMError, LLMUsage
from .profiles import Profile
from .lang import ENGLISH, pack
from .prompt_builder import ANYTHING_ELSE_QUESTION, build_directive, incomplete_closing
from .slots import (
    ANNOUNCE,
    ANSWER,
    COLLECT,
    FINAL,
    OUTCOME_INCOMPLETE,
    OUTCOME_NOT_FIT,
    OUTCOME_QUALIFIED,
    Plan,
    SlotValue,
    affects_outcome,
    compute_outcome,
    empty_slots,
    make_plan,
    merge_slots,
    plans_consistent,
    reply_schema,
)
from .speech import (
    EAGER_END_OF_TURN,
    END_OF_TURN,
    START_OF_TURN,
    TURN_RESUMED,
    SpeechEvent,
    SttError,
    SttInfo,
    TTSError,
    TTSRateLimited,
    TurnEvent,
)
from .stt_deepgram import AudioClock
from .text_utils import SentenceSplitter, clean_for_speech

log = logging.getLogger("voice_agent.agent")

#: English wording of the program's own phrases; other languages come from app/lang.py.
APOLOGY_TEXT = ENGLISH.apology
#: Said (from the phrase cache) when the text-to-speech service asks us to slow down.
ONE_MOMENT_TEXT = ENGLISH.one_moment
#: Longest we keep the caller waiting before the one retry after a rate limit.
MAX_RATE_LIMIT_WAIT_S = 4.0
MAX_HISTORY_MESSAGES = 24
GREETING_TIMEOUT_S = 5.0
DRAIN_TIMEOUT_S = 30.0
#: Longest the caller waits in silence while their goodbye sentence is read for slot values.
FINAL_EXTRACT_TIMEOUT_S = 2.5
#: Marker: these words end the call, but read them for slot values before choosing the closing.
_EXTRACT_FIRST = "extract-first"


def hangup_reason(outcome: str) -> str:
    if outcome == OUTCOME_NOT_FIT:
        return "not-a-fit"
    if outcome == OUTCOME_INCOMPLETE:
        return "caller-goodbye"
    return "goal-complete"  # qualified, or a profile-defined variant of it
def asks_anything_else(text: str, language: str = "en") -> bool:
    """Any wording of "is there anything else?" - only the server may ask that."""
    return pack(language).asks_anything_else(text)


def split_sentences(text: str) -> list[str]:
    splitter = SentenceSplitter()
    return splitter.feed(text + " ") + splitter.flush()


def goodbye_text(profile: Profile) -> str:
    """Spoken when the call hits the profile's hard time limit."""
    return pack(profile.language).time_limit.format(business=profile.business_name)


def closing_text(profile: Profile, outcome: str) -> str:
    """The closing line for an outcome. Chosen by code, never by the model."""
    if outcome == OUTCOME_QUALIFIED:
        return profile.closing_qualified
    if outcome == OUTCOME_NOT_FIT:
        return profile.closing_not_fit
    for variant in profile.outcomes:
        if variant.name == outcome:
            return variant.closing
    return incomplete_closing(profile)


def fixed_phrases(profile: Profile) -> list[str]:
    """Everything worth pre-generating for a profile."""
    return [
        profile.greeting,
        pack(profile.language).apology,
        pack(profile.language).one_moment,
        pack(profile.language).anything_else,
        profile.closing_qualified,
        profile.closing_not_fit,
        *(variant.closing for variant in profile.outcomes),
        incomplete_closing(profile),
        goodbye_text(profile),
    ]


class LLM(Protocol):
    async def reply_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        usage: Optional[LLMUsage] = None,
        *,
        text_fallback: bool = True,
    ) -> dict[str, Any]: ...


class TTS(Protocol):
    def stream(self, text: str) -> AsyncIterator[bytes]: ...


class Phrases(Protocol):
    def cached(self, text: str) -> Optional[bytes]: ...
    async def get(self, text: str) -> bytes: ...


class VoiceDetector(Protocol):
    last_rms: float

    def feed(self, frame: bytes) -> bool: ...
    def reset(self) -> None: ...


@dataclass
class TurnMetrics:
    """Latency of one caller turn -> bot reply, in milliseconds.

    * ``stt_ms``: caller's last word -> final transcript (EndOfTurn) received.
    * ``llm_first_token_ms``: LLM request sent -> reply received. Structured
      replies arrive in one piece, so this is the whole LLM time (including a
      second request when the first reply no longer fitted the plan).
    * ``tts_first_audio_ms``: first sentence sent to TTS -> first audio back.
    * ``total_ms``: caller's last word -> first reply frame sent to Asterisk.
      This is the pause the caller actually experiences (minus the phone network).

    With an eager head start the LLM/TTS stages begin before ``stt_ms`` is
    over, so the parts overlap and do not add up to the total.
    """

    index: int
    speech_end_wall: Optional[float]
    stt_ms: Optional[float]
    eager: bool = False
    llm_first_token_ms: Optional[float] = None
    tts_first_audio_ms: Optional[float] = None
    total_ms: Optional[float] = None
    #: How the model's reply was obtained: json, function_call or text_fallback
    #: (joined with "+" when the turn needed two requests).
    llm_path: Optional[str] = None


@dataclass(eq=False)
class _Job:
    """One reply being produced (LLM -> sentences -> TTS -> audio)."""

    transcript: str
    history: list[dict[str, str]]  # conversation including the caller's new message
    committed: bool  # False = speculative head start; audio is held back
    metrics: Optional[TurnMetrics] = None
    task: Optional["asyncio.Task[None]"] = None
    held: list[bytes] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)
    raw: list[dict[str, Any]] = field(default_factory=list)  # every structured reply, as received
    cancelled: bool = False
    finished: bool = False
    failed: bool = False
    llm_first_token_ms: Optional[float] = None
    tts_first_audio_ms: Optional[float] = None
    # what this reply will change once it is really said
    plan: Optional[Plan] = None
    slots_after: Optional[dict[str, Optional[SlotValue]]] = None
    rejected: list[str] = field(default_factory=list)
    replanned: bool = False
    dropped_sentences: int = 0
    #: How each reply of this job was obtained (json / function_call / text_fallback).
    paths: list[str] = field(default_factory=list)

    @property
    def llm_path(self) -> Optional[str]:
        return "+".join(self.paths) or None

    @property
    def spoke_plain_text(self) -> bool:
        """The reply being spoken came through the plain-text fallback: no slots were read."""
        return bool(self.paths) and self.paths[-1] == PATH_TEXT


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 1)


def with_user_message(history: list[dict[str, str]], text: str) -> list[dict[str, str]]:
    """History plus a user message; consecutive user messages are merged into one."""
    if history and history[-1]["role"] == "user":
        return history[:-1] + [{"role": "user", "content": f"{history[-1]['content']} {text}"}]
    return history + [{"role": "user", "content": text}]


class Agent:
    def __init__(
        self,
        *,
        profile: Profile,
        system_prompt: str,
        llm: LLM,
        tts: TTS,
        phrases: Phrases,
        pacer: Pacer,
        clock: AudioClock,
        record: Callable[[dict[str, Any]], None],
        hangup: Callable[[str], None],
        eager: bool = True,
        eager_allowed: Callable[[], bool] = lambda: True,
        merge_turns: bool = False,
        detector: Optional[VoiceDetector] = None,
        barge_confirm_s: float = 2.5,
        end_silence_s: float = 1.5,
        end_silence_max_s: float = 3.0,
        end_hold_max_s: float = 5.0,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._profile = profile
        #: The program's own phrases (apology, "one moment", ...) in the profile's language.
        self._lang = pack(profile.language)
        self._defs = profile.slots
        self._variants = profile.outcomes
        self._schema = reply_schema(profile.slots)
        self._system_prompt = system_prompt
        self._llm = llm
        self._tts = tts
        self._phrases = phrases
        self._pacer = pacer
        self._clock = clock
        self._record = record
        self._hangup = hangup
        self._eager = eager
        self._eager_allowed = eager_allowed
        #: Speech-to-text that ends a turn on a plain pause (Nova-3) cuts callers
        #: off between sentences. With this on, a caller who goes on speaking
        #: before the reply is audible gets both parts answered as one turn.
        self._merge_turns = merge_turns
        self._last_end_of_turn: Optional[tuple[float, str]] = None  # (when, what was said)
        self._merge: Optional[dict[str, Any]] = None
        self.turn_merges = 0
        self.eager_skipped = 0
        self._detector = detector
        self._barge_confirm_s = barge_confirm_s
        self._end_silence_s = end_silence_s
        #: Voice or noise after the closing line can keep the line open, but never longer than this in total.
        self._end_silence_max_s = max(end_silence_s, end_silence_max_s)
        self._now = now

        # ---- call state (the single source of truth for the flow)
        self.slots: dict[str, Optional[SlotValue]] = empty_slots(self._defs)
        #: The outcome the caller has been told about (and asked "anything else?" after).
        self.announced: Optional[str] = None
        #: Set when the closing line is spoken; cleared if the caller cancels the hang-up.
        self.outcome: Optional[str] = None

        self.history: list[dict[str, str]] = []
        self.turns: list[TurnMetrics] = []
        self.barge_ins = 0
        self.false_pauses = 0
        self.errors = 0
        self.tts_chars = 0
        self.replans = 0
        self.json_fallbacks = 0
        self.tts_rate_limited = 0
        #: Set when a reply of this call came from the fallback model (main model's daily limit reached).
        self.llm_fallback_model: Optional[str] = None
        #: How many replies came through each path (see GroqLLM.reply_json).
        self.llm_paths: dict[str, int] = {PATH_JSON: 0, PATH_FUNCTION: 0, PATH_TEXT: 0}
        #: Set after a plain-text fallback turn that was asking for a slot which
        #: decides the outcome: that slot is settled first on the next turn.
        self._recheck: Optional[str] = None
        self.llm_usage = LLMUsage()

        self._job: Optional[_Job] = None
        self._awaiting_first_frame: Optional[tuple[TurnMetrics, Optional[_Job]]] = None
        self._turn_index = 0
        self._closing = False  # True only once we are really hanging up
        self._background: set[asyncio.Task[None]] = set()
        # local barge-in pause
        self._paused_at: Optional[float] = None
        self._pause_timer: Optional[asyncio.TimerHandle] = None
        # cancellable ending
        self._end_task: Optional[asyncio.Task[None]] = None
        #: Identity of the pending hang-up. Cancelling replaces it, and the hang-up
        #: coroutine re-checks it after every wait. Task cancellation alone is not
        #: enough: asyncio.wait_for can swallow a cancel that lands at the very
        #: moment the awaited thing completes (e.g. the caller interrupts the
        #: closing line just as the audio queue empties).
        self._end_token: Optional[object] = None
        self._end_deadline = 0.0
        self._end_wait_started = 0.0  # 0 = not in the post-closing silence wait
        #: The caller started saying something during / after the closing line.
        #: The line keeps playing, but the hang-up waits (until this time at most)
        #: for the transcript, which decides whether it was a real new request.
        self._end_hold_until = 0.0
        self._end_hold_max_s = end_hold_max_s

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Start the pacer, play the greeting, arm the call-length limit."""
        self._pacer.start()
        self._spawn(self._enforce_max_duration())
        try:
            audio = await asyncio.wait_for(self._phrases.get(self._profile.greeting), GREETING_TIMEOUT_S)
        except (TTSError, asyncio.TimeoutError) as exc:
            self._error("tts", f"greeting unavailable: {exc or type(exc).__name__}")
            return
        if self._closing or self.history:
            return  # the caller spoke or hung up before the greeting was ready
        self._say_fixed(self._profile.greeting, audio, kind="greeting")

    async def close(self) -> None:
        """Call is over: stop everything. Safe to call more than once."""
        self._closing = True
        self._clear_pause()
        if self._job is not None:
            self._cancel_job(self._job, interrupted=False)
        tasks = list(self._background)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._pacer.stop()

    def _spawn(self, coro: Any) -> "asyncio.Task[None]":
        task = asyncio.get_running_loop().create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    # ------------------------------------------------------------------ caller audio (local barge-in)
    def on_caller_audio(self, frame: bytes) -> None:
        """Called for every inbound 20 ms frame. Cheap; never raises."""
        detector = self._detector
        if detector is None or self._closing:
            return
        try:
            if self._end_task is not None and self._pacer.speaking:
                detector.reset()  # the closing line is never paused on sound alone
            elif self._pacer.speaking and not self._pacer.paused:
                if detector.feed(frame):
                    self._local_pause(detector.last_rms)
                    detector.reset()
            elif self._end_task is not None and not self._pacer.speaking:
                # Closing line has been said; any voice keeps the line open a bit longer.
                if detector.feed(frame):
                    self._extend_ending(detector.last_rms)
                    detector.reset()
            else:
                detector.reset()
        except Exception:
            log.exception("voice detector failed; disabling it for this call")
            self._detector = None

    def _extend_ending(self, rms: float) -> None:
        """Sound on the line after the closing line: wait a little longer, up to the cap."""
        if not self._end_wait_started:
            return  # the closing line has not finished playing yet
        now = time.monotonic()
        limit = self._end_wait_started + self._end_silence_max_s
        deadline = min(now + self._end_silence_s, limit)
        if deadline <= self._end_deadline:
            return
        self._end_deadline = deadline
        self._record(
            {
                "type": "hangup_extended",
                "rms": round(rms),
                "waited_ms": round((now - self._end_wait_started) * 1000.0),
                "hangup_in_ms": round((deadline - now) * 1000.0),
                "capped": deadline >= limit,
            }
        )

    def _local_pause(self, rms: float) -> None:
        """The caller seems to be talking over the bot: go quiet now, decide later."""
        self._pacer.pause()
        self._paused_at = self._now()
        self._record({"type": "barge_pause", "rms": round(rms)})
        self._pause_timer = asyncio.get_running_loop().call_later(self._barge_confirm_s, self._resume_unconfirmed)

    def _resume_unconfirmed(self) -> None:
        """Speech-to-text never confirmed a turn: it was noise or echo. Carry on."""
        self._pause_timer = None
        if self._paused_at is None:
            return
        paused_ms = (self._now() - self._paused_at) * 1000.0
        self._paused_at = None
        self.false_pauses += 1
        self._pacer.resume()
        self._record({"type": "barge_resume", "paused_ms": round(paused_ms)})

    def _clear_pause(self) -> Optional[float]:
        """Forget a pending local pause; returns how long the bot had been paused (ms)."""
        if self._pause_timer is not None:
            self._pause_timer.cancel()
            self._pause_timer = None
        paused_ms = None if self._paused_at is None else (self._now() - self._paused_at) * 1000.0
        self._paused_at = None
        self._pacer.resume()
        return paused_ms

    # ------------------------------------------------------------------ speech-to-text events
    def on_stt_event(self, event: SpeechEvent, arrival_wall: float) -> None:
        """Entry point for every speech-to-text event. Synchronous and never raises."""
        try:
            if isinstance(event, SttError):
                self._error("stt", event.message)
            elif isinstance(event, SttInfo):
                if event.type == "Warning":
                    log.warning("speech-to-text warning: %s", event.detail)
            elif isinstance(event, TurnEvent) and not self._closing:
                self._record_stt_event(event, arrival_wall)
                if self._end_task is not None:
                    self._on_event_during_closing(event, arrival_wall)
                elif event.event == START_OF_TURN:
                    self._on_start_of_turn()
                elif event.event == EAGER_END_OF_TURN:
                    self._on_eager_end_of_turn(event)
                elif event.event == TURN_RESUMED:
                    self._on_turn_resumed()
                elif event.event == END_OF_TURN:
                    self._on_end_of_turn(event, arrival_wall)
        except Exception:
            log.exception("agent failed to handle a speech event")

    def _record_stt_event(self, event: TurnEvent, arrival_wall: float) -> None:
        """Keep every Flux turn event, with its audio timestamps, for diagnosis."""
        heard = self._clock.wall_time_at(event.audio_window_end)
        self._record(
            {
                "type": "stt_event",
                "event": event.event,
                "turn_index": event.turn_index,
                "transcript": event.transcript,
                "audio_start": round(event.audio_window_start, 3),
                "audio_end": round(event.audio_window_end, 3),
                "last_word_end": None if event.last_word_end is None else round(event.last_word_end, 3),
                "eot_confidence": round(event.eot_confidence, 3),
                # how long after we received that audio speech-to-text told us about it
                "lag_ms": None if heard is None else round(max(0.0, (arrival_wall - heard) * 1000.0)),
                "bot_speaking": self._pacer.speaking,
                **({"details": event.details} if event.details else {}),
            }
        )

    def _on_event_during_closing(self, event: TurnEvent, arrival_wall: float) -> None:
        """The caller makes a sound while the closing line plays or just after it.

        Sound alone changes nothing: the line is not paused, cut or restarted.
        The hang-up only waits until we know what was said. "Bye", "thanks",
        "ok" or a remark to someone else is ignored and the call ends as
        planned; a real new request (a question, "wait", "one more thing")
        stops the closing and is answered like any other turn.
        """
        if event.event == START_OF_TURN:
            self._end_hold_until = time.monotonic() + self._end_hold_max_s
            return
        if event.event != END_OF_TURN:
            return  # no head starts while closing
        self._end_hold_until = 0.0
        text = event.transcript
        if not text:
            return
        if interrupts_closing(text, self._profile.language) and self._ends_the_call(text) is None:
            self._cancel_ending()
            self._on_end_of_turn(event, arrival_wall)  # clears the rest of the closing audio
            return
        self._record({"type": "user", "turn": None, "text": text, "ignored": "closing"})

    def _on_start_of_turn(self) -> None:
        job = self._job
        if self._pacer.speaking:
            # Barge-in: the caller talks over the bot. Stop talking immediately.
            dropped = self._pacer.clear()
            paused_ms = self._clear_pause()
            if job is not None:
                self._cancel_job(job, interrupted=True)
            self.barge_ins += 1
            self._record({"type": "barge_in", "dropped_frames": dropped, "local_pause_ms": _round(paused_ms)})
        elif job is not None and job.committed:
            # We were still preparing a reply and the caller went on talking:
            # drop that reply (nothing of it was heard). The conversation history
            # joins the two parts, so the next reply answers both together.
            self._cancel_job(job, interrupted=False)
            if self._merge_turns and self._last_end_of_turn is not None and self._merge is None:
                ended, first = self._last_end_of_turn
                self._merge = {"first": first, "gap_ms": round((time.monotonic() - ended) * 1000.0)}

    def _ends_the_call(self, text: str) -> Optional[str]:
        """If these words from the caller end the call, the outcome to close with.

        * After the outcome was announced and "anything else?" asked: a decline
          or a goodbye closes with that outcome.
        * Before that, only an explicit goodbye ends the call. A bare "no" is
          just an answer to whatever was asked. The goodbye sentence may also
          carry an answer ("Yes. Thank you. Goodbye."), so the outcome is not
          known yet: ``_EXTRACT_FIRST`` is returned and the sentence is read
          for slot values before the closing line is chosen.
        """
        kind = caller_reply_kind(text, self._profile.language)
        if kind is None:
            return None
        plan = self._plan()
        if plan.kind == ANSWER:
            return plan.outcome
        if kind == GOODBYE:
            return _EXTRACT_FIRST
        return None

    def _plan(self, slots: Optional[dict[str, Optional[SlotValue]]] = None, announced: Optional[str] = None) -> Plan:
        return make_plan(self._defs, self.slots if slots is None else slots, announced or self.announced, self._variants)

    def _outcome(self, slots: Optional[dict[str, Optional[SlotValue]]] = None) -> Optional[str]:
        return compute_outcome(self._defs, self.slots if slots is None else slots, self._variants)

    def _on_eager_end_of_turn(self, event: TurnEvent) -> None:
        if not self._eager or not event.transcript:
            return
        if self._ends_the_call(event.transcript) is not None:
            return  # the closing line is a fixed phrase; no LLM head start needed
        if not self._eager_allowed():
            # Head starts cost extra LLM requests; skip them while the token budget is low.
            if self.eager_skipped == 0:
                self._record({"type": "eager_skipped", "reason": "token_budget_low"})
            self.eager_skipped += 1
            return
        job = self._job
        if job is not None:
            if job.committed:
                return
            if job.transcript == event.transcript and not job.failed:
                return  # same guess as before: keep the head start that is already running
            self._cancel_job(job, interrupted=False)
        self._start_job(event.transcript, committed=False)

    def _on_turn_resumed(self) -> None:
        job = self._job
        if job is not None and not job.committed:
            self._cancel_job(job, interrupted=False)

    def _on_end_of_turn(self, event: TurnEvent, arrival_wall: float) -> None:
        text = event.transcript
        job = self._job
        if not text:
            if job is not None and not job.committed:
                self._cancel_job(job, interrupted=False)
            return
        speech_end_wall = self._clock.wall_time_at(event.speech_end)
        stt_ms = None if speech_end_wall is None else max(0.0, (arrival_wall - speech_end_wall) * 1000.0)
        self._turn_index += 1
        metrics = TurnMetrics(self._turn_index, speech_end_wall, stt_ms)
        self._record(
            {
                "type": "user",
                "turn": metrics.index,
                "text": text,
                "stt_ms": _round(stt_ms),
                "eot_confidence": round(event.eot_confidence, 3),
            }
        )
        if self._merge is not None:
            merge, self._merge = self._merge, None
            self.turn_merges += 1
            self._record(
                {
                    "type": "turn_merge",
                    "turn": metrics.index,
                    "gap_ms": merge["gap_ms"],
                    "parts": [merge["first"], text],
                    "text": f"{merge['first']} {text}",
                }
            )
        self._last_end_of_turn = (time.monotonic(), text)
        if self._pacer.speaking:  # stale audio must never overlap a new reply
            self._pacer.clear()
        self._clear_pause()

        closing_outcome = self._ends_the_call(text)
        if closing_outcome is not None:
            if job is not None:
                self._cancel_job(job, interrupted=False)
            if closing_outcome == _EXTRACT_FIRST:
                # Goodbye before the outcome was announced: read the sentence for
                # slot values first, then compute the outcome and close.
                self._start_job(text, committed=True, metrics=metrics, final=True)
                self.history = with_user_message(self.history, text)
                return
            self.history = with_user_message(self.history, text)
            self._close_call(closing_outcome, metrics)
            return

        reusable = (
            job is not None
            and not job.committed
            and not job.failed
            and not job.cancelled
            and job.transcript == text
        )
        if reusable and job is not None:
            self.history = with_user_message(self.history, text)
            metrics.eager = True
            self._commit(job, metrics)
        else:
            if job is not None:
                self._cancel_job(job, interrupted=False)
            new_job = self._start_job(text, committed=True, metrics=metrics)
            self.history = with_user_message(self.history, text)
            self._awaiting_first_frame = (metrics, new_job)

    # ------------------------------------------------------------------ reply pipeline
    def _start_job(
        self, text: str, *, committed: bool, metrics: Optional[TurnMetrics] = None, final: bool = False
    ) -> _Job:
        history = with_user_message(self.history, text)[-MAX_HISTORY_MESSAGES:]
        job = _Job(transcript=text, history=history, committed=committed, metrics=metrics)
        self._job = job
        job.task = self._spawn(self._run_final_job(job) if final else self._run_job(job))
        return job

    async def _run_final_job(self, job: _Job) -> None:
        """The caller said goodbye before the outcome was announced.

        Their last sentence is read for slot values (nothing is spoken), then the
        code computes the outcome and plays the closing line that goes with it.
        If the model cannot be reached in time the call closes on what is known.
        """
        slots = dict(self.slots)
        started = self._now()
        try:
            reply = await asyncio.wait_for(
                # Only the slots matter here, so a plain-text reply is of no use.
                self._llm.reply_json(
                    self._messages(job, Plan(FINAL), slots, replanned=False),
                    self._schema,
                    self.llm_usage,
                    text_fallback=False,
                ),
                FINAL_EXTRACT_TIMEOUT_S,
            )
            job.raw.append(reply)
            job.paths.append(str(reply.get("path") or PATH_JSON))
            self._note_model(reply)
            slots, job.rejected = merge_slots(self._defs, slots, reply.get("slots"))
        except asyncio.CancelledError:
            raise
        except (LLMError, asyncio.TimeoutError) as exc:
            self._error("llm", f"could not read the caller's last sentence for slot values: {exc or type(exc).__name__}")
        except Exception as exc:  # never let a bug take the call down
            log.exception("final extraction crashed")
            self._error("agent", type(exc).__name__)
        if job.cancelled:
            return  # the caller went on talking
        job.llm_first_token_ms = (self._now() - started) * 1000.0
        job.finished = True
        if self._job is job:
            self._job = None
        self.slots = slots
        if job.raw:
            self._record_raw(job, job.metrics.index if job.metrics else None, complete=True)
        self._close_call(self._outcome() or OUTCOME_INCOMPLETE, job.metrics, job)

    def _messages(
        self, job: _Job, plan: Plan, slots: dict[str, Optional[SlotValue]], *, replanned: bool, recheck: bool = False
    ) -> list[dict[str, str]]:
        # Static prompt first, conversation next, this turn's instruction last:
        # the prefix never changes, so it stays cacheable.
        return (
            [{"role": "system", "content": self._system_prompt}]
            + job.history
            + [{"role": "system", "content": build_directive(plan, slots, replanned=replanned, recheck=recheck)}]
        )

    async def _run_job(self, job: _Job) -> None:
        try:
            # Snapshot: a speculative job must not see or change live state.
            slots_before = dict(self.slots)
            announced = self.announced
            plan = make_plan(self._defs, slots_before, announced, self._variants)
            # The last turn was answered without reading any slots, and it was asking
            # for a slot that decides the outcome: settle that slot before moving on.
            recheck = plan.kind == COLLECT and plan.target is not None and plan.target.name == self._recheck
            if recheck:
                plan = replace(plan, next=None)  # any progress then needs a fresh reply
            started = self._now()
            reply = await self._llm.reply_json(
                self._messages(job, plan, slots_before, replanned=False, recheck=recheck), self._schema, self.llm_usage
            )
            job.raw.append(reply)
            job.paths.append(str(reply.get("path") or PATH_JSON))
            self._note_model(reply)
            merged, job.rejected = merge_slots(self._defs, slots_before, reply.get("slots"))
            plan_after = make_plan(self._defs, merged, announced, self._variants)
            if not plans_consistent(plan, plan_after):
                # What the caller said changed what this turn must do (outcome decided,
                # or several things answered at once). The first reply was written
                # under the old plan, so get one for the new plan. Once only.
                job.replanned = True
                self.replans += 1
                plan = plan_after
                reply = await self._llm.reply_json(self._messages(job, plan, merged, replanned=True), self._schema, self.llm_usage)
                job.raw.append(reply)
                job.paths.append(str(reply.get("path") or PATH_JSON))
                self._note_model(reply)
                merged, more = merge_slots(self._defs, merged, reply.get("slots"))
                job.rejected += more
            job.llm_first_token_ms = (self._now() - started) * 1000.0
            job.plan = plan
            job.slots_after = merged

            sentences = [clean_for_speech(s) for s in split_sentences(str(reply.get("say") or ""))]
            sentences = [s for s in sentences if s and any(ch.isalnum() for ch in s)]
            # "Anything else?" belongs to the server. Drop it if the model says it anyway.
            kept = [s for s in sentences if not self._lang.asks_anything_else(s)]
            if not sentences:
                raise LLMError("the model returned an empty reply")
            if not kept and plan.kind != ANNOUNCE:
                kept = sentences  # never leave the caller with silence
            job.dropped_sentences = len(sentences) - len(kept)
            if getattr(self._tts, "one_request_per_reply", False):
                # Services with a tight request limit get the whole reply at once.
                kept = [" ".join(kept)] if kept else []
            for sentence in kept:
                await self._speak(job, sentence)
            if plan.kind == ANNOUNCE:
                # Outcome just decided: the server asks, once, whether there is anything else.
                await self._speak(job, self._lang.anything_else, count_chars=False)
        except asyncio.CancelledError:
            raise
        except (LLMError, TTSError) as exc:
            self._job_failed(job, "llm" if isinstance(exc, LLMError) else "tts", str(exc))
            return
        except Exception as exc:  # never let a bug take the call down
            log.exception("reply pipeline crashed")
            self._job_failed(job, "agent", type(exc).__name__)
            return
        job.finished = True
        if job.committed and not job.cancelled:
            self._finalize(job)

    def _note_model(self, reply: dict[str, Any]) -> None:
        """Log, once per call, that replies are coming from the fallback model."""
        model = reply.get("model")
        if model and self.llm_fallback_model is None:
            self.llm_fallback_model = str(model)
            self._record({"type": "llm_fallback_model", "turn": self._turn_index, "model": str(model)})

    async def _speak(self, job: _Job, text: str, *, count_chars: bool = True) -> None:
        """Turn one sentence into audio (cached phrase or TTS) and route it."""
        job.sentences.append(text)
        cached = self._phrases.cached(text)
        if cached is not None:
            if job.tts_first_audio_ms is None:
                job.tts_first_audio_ms = 0.0
            self._emit(job, cached)
            return
        if count_chars:
            self.tts_chars += len(text)
        started = self._now()
        try:
            await self._synthesize(job, text, started)
        except TTSRateLimited as exc:
            # Too many requests in a short time. Tell the caller we are still here,
            # wait a moment and try exactly once more; a second refusal is an error.
            wait_s = min(max(exc.retry_after_s, 0.0), MAX_RATE_LIMIT_WAIT_S)
            self.tts_rate_limited += 1
            self._record({"type": "tts_rate_limited", "retry_after_s": round(exc.retry_after_s, 1), "waiting_s": round(wait_s, 1)})
            moment = self._phrases.cached(self._lang.one_moment)
            if moment is not None:
                self._emit(job, moment)
            await asyncio.sleep(wait_s)
            if job.cancelled:
                return
            await self._synthesize(job, text, started)

    async def _synthesize(self, job: _Job, text: str, started: float) -> None:
        async for chunk in self._tts.stream(text):
            if job.tts_first_audio_ms is None:
                job.tts_first_audio_ms = (self._now() - started) * 1000.0
            if job.cancelled:
                return
            self._emit(job, chunk)

    def _emit(self, job: _Job, audio: bytes) -> None:
        if job.cancelled:
            return
        if job.committed:
            self._pacer.push(audio)
        else:
            job.held.append(audio)

    def _commit(self, job: _Job, metrics: TurnMetrics) -> None:
        """The head start was right: release the speculative reply."""
        job.committed = True
        job.metrics = metrics
        self._awaiting_first_frame = (metrics, job)
        for chunk in job.held:
            self._pacer.push(chunk)
        job.held.clear()
        if job.finished:
            self._finalize(job)

    def _apply_state(self, job: _Job, *, announced: bool) -> None:
        """Make the facts (and, if fully said, the announcement) of a reply official."""
        if job.slots_after is not None:
            self.slots = job.slots_after
        if announced and job.plan is not None and job.plan.kind == ANNOUNCE:
            self.announced = job.plan.outcome
        plan = job.plan
        if (
            job.spoke_plain_text
            and plan is not None
            and plan.kind == COLLECT
            and plan.target is not None
            and affects_outcome(plan.target, self._variants)
        ):
            self._recheck = plan.target.name  # see _run_job
        elif job.paths:
            self._recheck = None
        self._record_state(job)

    def _record_state(self, job: Optional[_Job] = None) -> None:
        """Per-turn snapshot of the slots and the decision made from them."""
        plan = self._plan()
        self._record(
            {
                "type": "state",
                "turn": self._turn_index,
                "slots": dict(self.slots),
                "outcome": self._outcome(),
                "announced": self.announced,
                "phase": plan.kind,
                "next_slot": plan.target.name if plan.target is not None else None,
                "replanned": bool(job and job.replanned),
                "llm_path": job.llm_path if job else None,
                "recheck": self._recheck,
                "rejected": list(job.rejected) if job else [],
            }
        )

    def _finalize(self, job: _Job) -> None:
        """The whole reply has been generated and queued."""
        self._pacer.flush()
        text = " ".join(job.sentences)
        self.history.append({"role": "assistant", "content": text})
        turn = job.metrics.index if job.metrics else None
        self._record({"type": "bot", "turn": turn, "text": text})
        self._record_raw(job, turn, complete=True)
        if self._job is job:
            self._job = None
        self._apply_state(job, announced=True)

    def _record_raw(self, job: _Job, turn: Optional[int], *, complete: bool) -> None:
        self._record(
            {
                "type": "llm_raw",
                "turn": turn,
                "text": " ".join(json.dumps(r, ensure_ascii=False) for r in job.raw),
                "complete": complete,
                "replanned": job.replanned,
                "dropped_sentences": job.dropped_sentences,
            }
        )
        for path in job.paths:
            self.llm_paths[path] = self.llm_paths.get(path, 0) + 1
        if job.spoke_plain_text:
            # The model wrote plain sentences instead of JSON; they were clean, so they
            # were spoken. Slots are unchanged this turn.
            self.json_fallbacks += 1
            self._record({"type": "llm_json_fallback", "turn": turn, "text": str(job.raw[-1].get("say") or "")})

    def _cancel_job(self, job: _Job, *, interrupted: bool) -> None:
        job.cancelled = True
        job.held.clear()
        if job.task is not None:
            job.task.cancel()
        if self._awaiting_first_frame is not None and self._awaiting_first_frame[1] is job:
            self._awaiting_first_frame = None
        if self._job is job:
            self._job = None
        if job.committed and job.sentences and not job.finished and not job.failed:
            text = " ".join(job.sentences)
            self.history.append({"role": "assistant", "content": text})
            turn = job.metrics.index if job.metrics else None
            self._record({"type": "bot", "turn": turn, "text": text, "interrupted": interrupted})
            self._record_raw(job, turn, complete=False)
            # What the caller told us stays true; an announcement cut short is repeated next turn.
            self._apply_state(job, announced=False)

    def _job_failed(self, job: _Job, stage: str, message: str) -> None:
        job.failed = True
        if job.cancelled:
            return
        self._error(stage, message)
        if not job.committed:
            return  # a failed head start is silent; EndOfTurn will try again
        if self._awaiting_first_frame is not None and self._awaiting_first_frame[1] is job:
            self._awaiting_first_frame = None
        if self._job is job:
            self._job = None
        if job.sentences:
            self.history.append({"role": "assistant", "content": " ".join(job.sentences)})
        if job.slots_after is not None:
            self._apply_state(job, announced=False)
        audio = self._phrases.cached(self._lang.apology)
        if audio:
            self._say_fixed(self._lang.apology, audio, kind="apology")

    # ------------------------------------------------------------------ fixed phrases / endings
    def _say_fixed(self, text: str, audio: bytes, *, kind: str) -> None:
        self._pacer.push(audio)
        self._pacer.flush()
        self.history.append({"role": "assistant", "content": text})
        self._record({"type": "bot", "turn": None, "text": text, "kind": kind})

    def _error(self, stage: str, message: str) -> None:
        self.errors += 1
        log.error("%s error: %s", stage, message)
        self._record({"type": "error", "stage": stage, "message": message[:300]})

    def _close_call(self, outcome: str, metrics: Optional[TurnMetrics] = None, job: Optional[_Job] = None) -> None:
        """The caller declined or said goodbye: say the closing line the outcome calls for."""
        self._cancel_ending(silent=True)
        self.outcome = outcome
        text = closing_text(self._profile, outcome)
        if metrics is not None:
            self._awaiting_first_frame = (metrics, job)
        self._record_state(job)
        self._record(
            {"type": "ending", "outcome": outcome, "silence_s": self._end_silence_s, "max_s": self._end_silence_max_s}
        )
        token = object()
        self._end_token = token
        self._end_task = self._spawn(self._say_closing_then_hang_up(text, hangup_reason(outcome), token))

    async def _say_closing_then_hang_up(self, text: str, reason: str, token: object) -> None:
        audio = self._phrases.cached(text)
        try:
            if audio is None:
                audio = await self._phrases.get(text)  # not pre-generated yet: make it now
        except TTSError as exc:
            self._error("tts", f"closing line unavailable: {exc}")
            audio = b""
        if token is not self._end_token:
            return  # the caller spoke in the meantime
        if audio:
            self._say_fixed(text, audio, kind="closing")
        await self._hangup_when_done(reason, wait_for_silence=True, token=token)

    def _cancel_ending(self, *, silent: bool = False) -> None:
        """The caller spoke after the closing line: stay on the call."""
        task = self._end_task
        if task is None:
            return
        self._end_task = None
        self._end_token = None
        self._end_wait_started = 0.0
        self._end_hold_until = 0.0
        self.outcome = None
        task.cancel()
        if not silent:
            self._record({"type": "hangup_cancelled"})

    async def _enforce_max_duration(self) -> None:
        await asyncio.sleep(self._profile.max_call_seconds)
        if self._closing:
            return
        self._closing = True
        self._cancel_ending(silent=True)
        if self._job is not None:
            self._cancel_job(self._job, interrupted=True)
        self._pacer.clear()
        self._clear_pause()
        self._record({"type": "max_duration", "seconds": self._profile.max_call_seconds})
        text = goodbye_text(self._profile)
        audio = self._phrases.cached(text)
        if audio:
            self._say_fixed(text, audio, kind="goodbye")
        await self._hangup_when_done("max-duration", wait_for_silence=False)

    async def _hangup_when_done(self, reason: str, *, wait_for_silence: bool, token: Optional[object] = None) -> None:
        def still_wanted() -> bool:
            return token is None or token is self._end_token

        try:
            await asyncio.wait_for(self._pacer.drained(), DRAIN_TIMEOUT_S)
        except asyncio.TimeoutError:
            pass
        if not still_wanted():
            return
        if wait_for_silence:
            # Give the caller a moment to object. Sound on the line pushes the
            # deadline out (see _extend_ending), but only up to end_silence_max_s.
            self._end_wait_started = time.monotonic()
            self._end_deadline = self._end_wait_started + self._end_silence_s
            while True:
                now = time.monotonic()
                remaining = self._end_deadline - now
                if remaining <= 0:
                    if now < self._end_hold_until:
                        # The caller is in the middle of saying something: wait for
                        # the transcript before hanging up (see _on_event_during_closing).
                        await asyncio.sleep(0.05)
                        if not still_wanted():
                            return
                        continue
                    break
                await asyncio.sleep(remaining)
                if not still_wanted():
                    return
        else:
            await asyncio.sleep(0.4)
        if not still_wanted():
            return
        self._closing = True
        self._end_task = None
        self._end_wait_started = 0.0
        self._hangup(reason)

    # ------------------------------------------------------------------ pacer callback
    def on_frame_sent(self) -> None:
        """Called by the pacer for every frame; closes the latency measurement of a turn."""
        waiting = self._awaiting_first_frame
        if waiting is None:
            return
        m, job = waiting
        if job is not None and not job.committed:
            return
        self._awaiting_first_frame = None
        if job is not None:
            m.llm_first_token_ms = job.llm_first_token_ms
            m.tts_first_audio_ms = job.tts_first_audio_ms
            m.llm_path = job.llm_path
        if m.speech_end_wall is not None:
            m.total_ms = max(0.0, (self._now() - m.speech_end_wall) * 1000.0)
        self.turns.append(m)
        self._record(
            {
                "type": "turn",
                "turn": m.index,
                "stt_ms": _round(m.stt_ms),
                "llm_first_token_ms": _round(m.llm_first_token_ms),
                "tts_first_audio_ms": _round(m.tts_first_audio_ms),
                "total_ms": _round(m.total_ms),
                "eager": m.eager,
                "llm_path": m.llm_path,
            }
        )

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict[str, Any]:
        totals = [t.total_ms for t in self.turns if t.total_ms is not None]
        return {
            "turns": len(self.turns),
            "total_ms_avg": _round(sum(totals) / len(totals)) if totals else None,
            "total_ms_max": _round(max(totals)) if totals else None,
            "barge_ins": self.barge_ins,
            "false_pauses": self.false_pauses,
            "eager_skipped": self.eager_skipped,
            "outcome": self.outcome,
            "computed_outcome": self._outcome(),
            "slots": dict(self.slots),
            "replans": self.replans,
            "turn_merges": self.turn_merges,
            "json_fallbacks": self.json_fallbacks,
            "llm_paths": dict(self.llm_paths),
            "llm_fallback_model": self.llm_fallback_model,
            "errors": self.errors,
            "llm_requests": self.llm_usage.requests,
            "llm_tokens": self.llm_usage.total_tokens,
            "llm_prompt_tokens": self.llm_usage.prompt_tokens,
            "llm_completion_tokens": self.llm_usage.completion_tokens,
            "llm_cached_tokens": self.llm_usage.cached_tokens,
            "tts_chars": self.tts_chars,
            "tts_rate_limited": self.tts_rate_limited,
        }
