"""Shared fakes for agent tests: a scripted LLM, TTS and phrase cache, and a
:class:`Rig` that wires an :class:`~app.agent.Agent` to them with a real pacer."""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Callable, Optional

from app.agent import Agent
from app.audio_out import FRAME_BYTES, Pacer
from app.llm_groq import LLMUsage
from app.profiles import Profile
from app.slots import SlotDef
from app.stt_deepgram import AudioClock
from app.stt_flux import TurnEvent
from app.tts_aura import TTSError

FRAME = bytes(FRAME_BYTES)

OWNS_HOME = SlotDef(
    name="owns_home",
    type="yes_no",
    question="Do you own the house, or do you rent?",
    disqualifying=("no",),
    not_fit_reason="we only install on houses the caller owns",
    description="yes only if the caller owns a house; renting or an apartment is no",
)
BILL = SlotDef(name="monthly_bill", type="text", question="Roughly how much is your monthly electricity bill?")
NAME = SlotDef(name="name", type="text", question="What name should the advisor ask for?")

PROFILE = Profile(
    name="test",
    business_name="Acme Solar",
    agent_role="the receptionist",
    goal="Qualify the caller.",
    greeting="Hello, Acme Solar.",
    slots=(OWNS_HOME, BILL, NAME),
    knowledge="We install solar panels on owned houses.",
    tone="friendly",
    closing_qualified="Thanks, an advisor will call you back. Goodbye!",
    closing_not_fit="Sorry we cannot help with that. Goodbye!",
    max_call_seconds=300,
)


def reply(say: str, **slots: Any) -> dict[str, Any]:
    """A structured model reply."""
    return {"slots": dict(slots), "say": say}


class FakeLLM:
    """Returns scripted structured replies; can stall (request in flight) or fail."""

    def __init__(self, replies: Optional[list[Any]] = None) -> None:
        self.replies = list(replies or [])
        self.requests: list[list[dict[str, str]]] = []
        self.schemas: list[dict[str, Any]] = []
        self.cancelled = 0
        self.hold: Optional[asyncio.Event] = None  # when set: the request hangs until released

    async def reply_json(
        self, messages: list[dict[str, str]], schema: dict[str, Any], usage: Optional[LLMUsage] = None
    ) -> dict[str, Any]:
        self.requests.append(messages)
        self.schemas.append(schema)
        if usage is not None:
            usage.requests += 1
            usage.prompt_tokens += 100
            usage.completion_tokens += 10
        item = self.replies.pop(0) if self.replies else "Okay. Do you own the house?"
        try:
            if self.hold is not None:
                await self.hold.wait()
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if isinstance(item, BaseException):
            raise item
        return reply(item) if isinstance(item, str) else item

    def directive(self, index: int = -1) -> str:
        """The per-turn instruction the code gave the model in request ``index``."""
        return self.requests[index][-1]["content"]


class FakeTTS:
    """Returns ``frames`` frames of audio per sentence."""

    def __init__(self, frames_per_sentence: int = 2) -> None:
        self.frames = frames_per_sentence
        self.spoken: list[str] = []
        self.fail: Optional[BaseException] = None
        self.cancelled = 0

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        if self.fail is not None:
            raise self.fail
        self.spoken.append(text)
        try:
            for _ in range(self.frames):
                yield FRAME
                await asyncio.sleep(0)
        except (asyncio.CancelledError, GeneratorExit):
            self.cancelled += 1
            raise


class FakePhrases:
    """Phrase cache: texts in ``known`` are pre-generated; get() can synthesize any text."""

    def __init__(self, missing: bool = False, known: Optional[set[str]] = None) -> None:
        self.missing = missing
        self.known = known or set()
        self.served: list[str] = []

    def cached(self, text: str) -> Optional[bytes]:
        if self.missing or text not in self.known:
            return None
        self.served.append(text)
        return FRAME * 2

    async def get(self, text: str) -> bytes:
        if self.missing:
            raise TTSError("Aura HTTP 500")
        self.served.append(text)
        return FRAME * 2


def phrases_for(profile: Profile) -> FakePhrases:
    from app.agent import fixed_phrases

    return FakePhrases(known=set(fixed_phrases(profile)))


class Rig:
    """An Agent wired to fakes, with a real pacer running in real time."""

    def __init__(
        self,
        llm: Optional[FakeLLM] = None,
        tts: Optional[FakeTTS] = None,
        *,
        eager: bool = True,
        profile: Profile = PROFILE,
        phrases: Optional[FakePhrases] = None,
        detector: Any = None,
        barge_confirm_s: float = 0.15,
        end_silence_s: float = 0.05,
        end_silence_max_s: float = 1.0,
    ) -> None:
        self.llm = llm or FakeLLM()
        self.tts = tts or FakeTTS()
        self.phrases = phrases or phrases_for(profile)
        self.records: list[dict[str, Any]] = []
        self.frames: list[bytes] = []
        self.hangups: list[str] = []
        self.clock = AudioClock()
        base = time.time()
        for _ in range(500):  # 10 s of caller audio, all "heard" just now
            self.clock.mark(0.02, base)
        self.pacer = Pacer(self.frames.append, on_frame=lambda: self.agent.on_frame_sent())
        self.agent = Agent(
            profile=profile,
            system_prompt="SYSTEM",
            llm=self.llm,
            tts=self.tts,
            phrases=self.phrases,
            pacer=self.pacer,
            clock=self.clock,
            record=self.records.append,
            hangup=self.hangups.append,
            eager=eager,
            detector=detector,
            barge_confirm_s=barge_confirm_s,
            end_silence_s=end_silence_s,
            end_silence_max_s=end_silence_max_s,
        )

    def event(self, kind: str, text: str = "", end: float = 5.0) -> None:
        self.agent.on_stt_event(TurnEvent(kind, 0, text, 0.8, end, end), time.time())

    async def say(self, text: str) -> None:
        """A complete caller turn: start, end, and wait until the bot has answered."""
        self.event("StartOfTurn", text.split()[0] if text.split() else "")
        self.event("EndOfTurn", text)
        await settle(self)

    def kinds(self) -> list[str]:
        """Conversation-level records (diagnostic lines left out)."""
        return [r["type"] for r in self.records if r["type"] not in ("stt_event", "llm_raw", "state")]

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r["type"] == kind]

    def bot_lines(self) -> list[str]:
        return [r["text"] for r in self.of("bot")]


async def until(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.002)


async def settle(rig: Rig) -> None:
    """Wait until the bot has nothing left to say."""
    await until(lambda: rig.agent._job is None)
    await asyncio.wait_for(rig.pacer.drained(), 2)


def run(scenario: Callable[[], Any]) -> Any:
    return asyncio.run(scenario())
