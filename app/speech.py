"""The two seams of the voice pipeline: speech-to-text and text-to-speech.

The agent does not care *which* service hears the caller or speaks the reply.
It only needs:

* a :class:`STTProvider` - takes the caller's audio (20 ms frames of 8 kHz
  16-bit PCM) and reports **turn events**: the caller started speaking, the
  caller finished and this is what they said;
* a :class:`TTSProvider` - takes a sentence and yields audio in that same
  format, chunk by chunk, as the service produces it.

Each service gets a small adapter that implements one of these (Flux and Aura
today). :mod:`app.providers` picks the adapters named in the profile. Adding a
language is then a matter of adding adapters and a profile, not of changing
the agent.

Everything here is service-neutral on purpose: no URLs, no API keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional, Protocol, Union, runtime_checkable

# ---- turn events every STT adapter reports ---------------------------------
START_OF_TURN = "StartOfTurn"  # the caller began speaking (used for barge-in)
UPDATE = "Update"  # transcript so far; informational
EAGER_END_OF_TURN = "EagerEndOfTurn"  # probably finished: the agent may start thinking
TURN_RESUMED = "TurnResumed"  # no, they kept talking: discard that head start
END_OF_TURN = "EndOfTurn"  # finished; `transcript` is the whole turn


@dataclass(frozen=True)
class TurnEvent:
    """Something happened in the caller's turn. Times are seconds of caller audio."""

    event: str
    turn_index: int
    transcript: str
    eot_confidence: float = 0.0
    audio_window_end: float = 0.0
    audio_window_start: float = 0.0
    last_word_end: Optional[float] = None

    @property
    def speech_end(self) -> float:
        """Best estimate (audio seconds) of when the caller's last word ended."""
        return self.last_word_end if self.last_word_end is not None else self.audio_window_end


@dataclass(frozen=True)
class SttInfo:
    """Connected / Warning / other messages that are not about a turn."""

    type: str
    detail: str = ""


@dataclass(frozen=True)
class SttError:
    """Speech-to-text failed; the agent can no longer hear the caller. Safe to log."""

    message: str


SpeechEvent = Union[TurnEvent, SttInfo, SttError]
#: Called for every event with the wall-clock time it arrived. Must not raise.
SpeechEventHandler = Callable[[SpeechEvent, float], None]


class TTSError(Exception):
    """Speech synthesis failed. Message is safe to log."""


@runtime_checkable
class STTProvider(Protocol):
    """One speech-to-text stream = one call. Audio in, turn events out.

    Events go to the handler given when the stream is created. No method may
    raise into the call: a failure is reported once as :class:`SttError` and
    ``failed`` becomes true.
    """

    #: Short adapter name for logs, e.g. "flux".
    name: str
    #: The model in use, for logs.
    model: str
    failed: bool

    @property
    def supports_eager(self) -> bool:
        """Does this stream send EagerEndOfTurn / TurnResumed (speculative head starts)?"""
        ...

    def start(self) -> None:
        """Open the connection in the background. Returns immediately."""
        ...

    def send_audio(self, payload: bytes, arrival_wall: Optional[float] = None) -> None:
        """Queue one frame of caller audio. Never blocks, never raises."""
        ...

    async def close(self) -> None:
        """The call is over: flush and disconnect."""
        ...


@runtime_checkable
class TTSProvider(Protocol):
    """Text in, 8 kHz 16-bit mono PCM out. Shared by all calls."""

    #: Service name for logs and error messages, e.g. "Aura".
    name: str
    #: The voice (or voice model) that speaks.
    voice: str
    #: Identifies the sound of this provider. Cached phrase audio is stored
    #: under it, so a different voice never plays another voice's recording.
    cache_key: str
    #: A URL worth touching every few seconds during a call to keep the HTTPS
    #: connection open, or None if the service should not be pinged.
    keep_warm_url: Optional[str]

    def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield audio for ``text`` as it is generated. Raises :class:`TTSError`.

        Stopping the iteration early (the caller interrupted) must stop the synthesis.
        """
        ...

    async def synthesize(self, text: str) -> bytes:
        """The whole utterance at once (for phrases that are cached)."""
        ...
