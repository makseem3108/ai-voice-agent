"""The closing line is finished once it has started.

Real call 2026-11-12 (5G): the caller said "Okay. Thank you.", the closing line
began, 0.2 s later they added "Goodbye." - which cancelled the hang-up and
started the closing line again from the top."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import pytest

from app.call_log import format_record
from app.intent import caller_reply_kind, interrupts_closing
from tests.helpers import FRAME, PROFILE, FakeLLM, FakePhrases, Rig, phrases_for, reply, run, until

NOT_FIT = PROFILE.closing_not_fit
CLOSING_FRAMES = 15  # 0.3 s of closing audio, long enough to talk over


class LongPhrases(FakePhrases):
    """Like the normal fake, but fixed phrases last 0.3 s."""

    def cached(self, text: str) -> Optional[bytes]:
        audio = super().cached(text)
        return None if audio is None else FRAME * CLOSING_FRAMES


class Noise:
    last_rms = 3000.0

    def feed(self, frame: bytes) -> bool:
        return True

    def reset(self) -> None:
        pass


def not_fit_rig(**kwargs: Any) -> Rig:
    llm = FakeLLM([reply("Thanks.", owns_home="no"), reply("We only install on houses you own."), reply("The site visit is free.")])
    phrases = LongPhrases(known=phrases_for(PROFILE).known)
    return Rig(llm, phrases=phrases, **kwargs)


async def reach_the_closing_line(rig: Rig) -> int:
    """Not-a-fit announced, caller declines, the closing line has just started. Returns frames sent so far."""
    rig.pacer.start()
    await rig.say("I rent.")
    rig.event("StartOfTurn", "Okay")
    rig.event("EndOfTurn", "Okay. Thank you.")
    await until(lambda: rig.pacer.speaking and len(rig.of("ending")) == 1)
    return len(rig.frames)


# ====================================================================== the real call
@pytest.mark.parametrize("words", ["Goodbye.", "Thanks.", "Okay.", "Okay, thank you, bye.", "Yeah, let's go and eat something.", "Mhm."])
def test_real_call_6_talking_over_the_closing_line_neither_cuts_nor_restarts_it(words: str) -> None:
    async def scenario() -> tuple[Rig, int]:
        rig = not_fit_rig(end_silence_s=0.1)
        before = await reach_the_closing_line(rig)
        await asyncio.sleep(0.06)  # the caller speaks 60 ms into the line
        rig.event("StartOfTurn", words.split()[0])
        rig.event("EndOfTurn", words)
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig, before

    rig, before = run(scenario)
    assert rig.bot_lines().count(NOT_FIT) == 1  # said once, not restarted
    # ... and said to the end (the first frame may already have gone out when we started counting)
    assert CLOSING_FRAMES - 1 <= len(rig.frames) - before <= CLOSING_FRAMES
    assert rig.of("hangup_cancelled") == [] and rig.of("barge_in") == [] and rig.agent.barge_ins == 0
    assert len(rig.of("ending")) == 1 and rig.hangups == ["not-a-fit"]
    assert len(rig.llm.requests) == 2  # the remark never went to the model
    (ignored,) = [r for r in rig.of("user") if r.get("ignored")]
    assert ignored == {"type": "user", "turn": None, "text": words, "ignored": "closing"}
    assert "said during the closing line" in (format_record(ignored) or "")


def test_sound_alone_never_pauses_the_closing_line() -> None:
    """The local voice detector may pause ordinary replies, but not the closing line."""

    async def scenario() -> tuple[Rig, float]:
        rig = not_fit_rig(detector=Noise(), end_silence_s=0.05, end_silence_max_s=0.1)
        await reach_the_closing_line(rig)
        started = time.monotonic()
        while not rig.hangups:
            rig.agent.on_caller_audio(FRAME)  # constant voice-like noise on the line
            assert not rig.pacer.paused
            await asyncio.sleep(0.005)
        elapsed = time.monotonic() - started
        await rig.agent.close()
        return rig, elapsed

    rig, elapsed = run(scenario)
    assert rig.bot_lines().count(NOT_FIT) == 1
    after_ending = rig.records[rig.records.index(rig.of("ending")[0]):]
    assert [r for r in after_ending if r["type"] in ("barge_pause", "barge_resume", "barge_in")] == []
    assert elapsed < 0.3 + 0.1 + 0.25  # closing audio + capped wait + slack: noise cannot hold the line
    assert rig.hangups == ["not-a-fit"]


# ====================================================================== a real new request still gets through
@pytest.mark.parametrize("words", ["Wait, one more thing, is the visit free?", "Is the visit free?", "Hold on.", "Actually I do have a question."])
def test_a_real_question_stops_the_closing_line_and_is_answered(words: str) -> None:
    async def scenario() -> tuple[Rig, int]:
        rig = not_fit_rig(end_silence_s=0.1)
        before = await reach_the_closing_line(rig)
        await asyncio.sleep(0.06)
        rig.event("StartOfTurn", words.split()[0])
        assert rig.pacer.speaking and rig.of("hangup_cancelled") == []  # the sound alone changed nothing
        rig.event("EndOfTurn", words)  # the transcript decides
        await until(lambda: rig.agent._job is None and len(rig.llm.requests) == 3)
        await asyncio.wait_for(rig.pacer.drained(), 2)
        await asyncio.sleep(0.3)
        assert rig.hangups == [] and rig.agent.summary()["outcome"] is None
        await rig.say("No, that's all.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig, before

    rig, before = run(scenario)
    assert len(rig.of("hangup_cancelled")) == 1
    assert rig.bot_lines()[-3:] == [NOT_FIT, "The site visit is free.", NOT_FIT]
    assert rig.llm.requests[-1][-2]["content"].endswith(words)
    assert rig.hangups == ["not-a-fit"]


def test_hang_up_waits_for_the_transcript_of_what_the_caller_is_saying() -> None:
    """The closing audio and the silence window are over, but the caller is mid-sentence."""

    async def scenario(final_words: str) -> Rig:
        rig = not_fit_rig(end_silence_s=0.05, end_hold_max_s=2.0)
        await reach_the_closing_line(rig)
        await asyncio.sleep(0.25)  # near the end of the line
        rig.event("StartOfTurn", "And")
        await asyncio.sleep(0.4)  # well past audio end + silence window
        assert rig.hangups == []  # still listening
        rig.event("EndOfTurn", final_words)
        await asyncio.sleep(0.3)
        await rig.agent.close()
        return rig

    question = run(lambda: scenario("And what about batteries, do you install those?"))
    assert question.hangups == [] and len(question.of("hangup_cancelled")) == 1
    farewell = run(lambda: scenario("And thanks again, bye."))
    assert farewell.hangups == ["not-a-fit"] and farewell.of("hangup_cancelled") == []


def test_hang_up_does_not_wait_forever_for_a_transcript() -> None:
    async def scenario() -> tuple[Rig, float]:
        rig = not_fit_rig(end_silence_s=0.05, end_hold_max_s=0.3)
        await reach_the_closing_line(rig)
        rig.event("StartOfTurn", "Uh")  # and then nothing more from speech-to-text
        started = time.monotonic()
        await until(lambda: bool(rig.hangups), timeout=3.0)
        elapsed = time.monotonic() - started
        await rig.agent.close()
        return rig, elapsed

    rig, elapsed = run(scenario)
    assert rig.hangups == ["not-a-fit"] and rig.bot_lines().count(NOT_FIT) == 1
    assert 0.25 <= elapsed < 0.9


def test_a_goodbye_with_wait_in_it_does_not_restart_the_closing() -> None:
    async def scenario() -> Rig:
        rig = not_fit_rig(end_silence_s=0.1)
        await reach_the_closing_line(rig)
        rig.event("StartOfTurn", "Wait")
        rig.event("EndOfTurn", "Wait, never mind, bye.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines().count(NOT_FIT) == 1 and rig.of("hangup_cancelled") == []


# ====================================================================== the transcript rule
@pytest.mark.parametrize(
    "text",
    [
        "Wait.", "Hold on a second.", "One more thing.", "I have a question.", "Can I ask something else?", "What about batteries?",
        "How long is the warranty", "Actually, I own the house.", "Don't hang up.", "Tell me about the warranty.", "Is the visit free?",
        "No wait, one more question.",
    ],
)
def test_real_requests_interrupt_the_closing(text: str) -> None:
    assert interrupts_closing(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "", "Goodbye.", "Bye bye.", "Thanks.", "Thank you very much.", "Okay.", "Ok thanks, bye.", "No thanks.", "That's all.",
        "Have a nice day.", "You too.", "Yeah.", "Mhm.", "Great.", "Let's go and eat something.", "I'm coming in a minute.",
        "Thank you for calling Northwind Solar.",  # the bot's own echo
    ],
)
def test_farewells_acknowledgements_and_side_talk_do_not(text: str) -> None:
    assert interrupts_closing(text) is False


def test_dont_hang_up_is_not_a_goodbye() -> None:
    assert caller_reply_kind("Don't hang up.") is None and caller_reply_kind("Please don't hang up yet") is None
    assert caller_reply_kind("You can hang up now.") == "goodbye"
