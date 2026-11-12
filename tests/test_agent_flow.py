"""Agent mechanics that do not depend on the slot logic: turns, barge-in,
eager head start, failures, call length. (Slot filling and endings are in
test_slots.py.)"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import AsyncIterator

from app.agent import APOLOGY_TEXT, goodbye_text
from app.llm_groq import LLMError
from app.stt_deepgram import SttError
from app.stt_flux import FluxInfo
from app.tts_aura import TTSError
from tests.helpers import FRAME, PROFILE, FakeLLM, FakePhrases, FakeTTS, Rig, reply, run, settle, until


class SlowTTS(FakeTTS):
    """Audio keeps coming until the request is cancelled (a long sentence still being synthesized)."""

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        self.spoken.append(text)
        try:
            while True:
                yield FRAME
                await asyncio.sleep(0.005)
        except (asyncio.CancelledError, GeneratorExit):
            self.cancelled += 1
            raise


# ---------------------------------------------------------------------- basic turn
def test_greeting_is_played_first_and_remembered() -> None:
    async def scenario() -> Rig:
        rig = Rig()
        await rig.agent.start()
        await settle(rig)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert len(rig.frames) == 2
    assert rig.of("bot")[0] == {"type": "bot", "turn": None, "text": PROFILE.greeting, "kind": "greeting"}
    assert rig.agent.history == [{"role": "assistant", "content": PROFILE.greeting}]
    assert rig.llm.requests == []  # the greeting costs no LLM call


def test_a_turn_goes_stt_to_llm_to_tts_to_caller() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM([reply("Sure. Do you own the house, or do you rent?")]))
        await rig.agent.start()
        await settle(rig)
        await rig.say("Hi, I want solar.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.tts.spoken == ["Sure.", "Do you own the house, or do you rent?"]  # sentence by sentence
    assert len(rig.frames) == 2 + 4  # greeting + two sentences
    assert rig.kinds() == ["bot", "user", "turn", "bot"]
    request = rig.llm.requests[0]
    # static prompt first, conversation next, this turn's instruction last
    assert request[0] == {"role": "system", "content": "SYSTEM"}
    assert request[1:3] == [
        {"role": "assistant", "content": PROFILE.greeting},
        {"role": "user", "content": "Hi, I want solar."},
    ]
    assert request[-1]["role"] == "system" and request[-1]["content"].startswith("INSTRUCTION")
    assert len(request) == 4
    assert rig.agent.history[-1] == {"role": "assistant", "content": "Sure. Do you own the house, or do you rent?"}
    (turn,) = rig.of("turn")
    assert turn["turn"] == 1 and turn["eager"] is False
    for key in ("stt_ms", "llm_first_token_ms", "tts_first_audio_ms", "total_ms"):
        assert isinstance(turn[key], float) and turn[key] >= 0
    summary = rig.agent.summary()
    assert summary["turns"] == 1 and summary["llm_tokens"] == 110
    assert summary["tts_chars"] == len("Sure.") + len("Do you own the house, or do you rent?")


def test_only_say_is_spoken_never_the_slots_or_other_keys() -> None:
    async def scenario() -> Rig:
        answer = {"slots": {"owns_home": "yes", "name": None}, "say": "Great.", "note": "internal thought, never spoken"}
        rig = Rig(FakeLLM([answer]))
        rig.pacer.start()
        await rig.say("I own my house.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.tts.spoken == ["Great."]
    spoken = " ".join(rig.tts.spoken + rig.bot_lines())
    for leaked in ("owns_home", "slots", "internal thought", "{", "yes"):
        assert leaked not in spoken, leaked


def test_prompt_prefix_is_stable_across_requests() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM(["One.", "Two.", "Three."]))
        rig.pacer.start()
        for said in ("First.", "Second.", "Third."):
            await rig.say(said)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert [r[0] for r in rig.llm.requests] == [{"role": "system", "content": "SYSTEM"}] * 3
    # apart from the trailing instruction, each request only APPENDS to the previous one
    for earlier, later in zip(rig.llm.requests, rig.llm.requests[1:]):
        assert later[: len(earlier) - 1] == earlier[:-1]
    assert all(r[-1]["content"].startswith("INSTRUCTION") for r in rig.llm.requests)


# ---------------------------------------------------------------------- barge-in
def test_barge_in_stops_audio_and_cancels_tts_in_flight() -> None:
    async def scenario() -> tuple[Rig, int, int]:
        rig = Rig(FakeLLM([reply("Let me explain everything in detail. This part is never spoken.")]), SlowTTS())
        await rig.agent.start()
        await settle(rig)
        rig.event("EndOfTurn", "Tell me everything.")
        await until(lambda: len(rig.frames) >= 2 + 3)  # the bot is audibly talking
        assert rig.pacer.speaking
        rig.event("StartOfTurn", "Wait")  # caller talks over the bot
        at_barge_in = len(rig.frames)
        await asyncio.sleep(0.1)  # 5 frame periods: nothing more may be sent
        later = len(rig.frames)
        await until(lambda: rig.tts.cancelled == 1)
        await rig.agent.close()
        return rig, at_barge_in, later

    rig, at_barge_in, later = run(scenario)
    assert later - at_barge_in <= 1  # at most the frame already in flight
    assert rig.pacer.speaking is False
    assert rig.tts.cancelled == 1 and rig.agent.barge_ins == 1
    assert len(rig.of("barge_in")) == 1
    assert rig.tts.spoken == ["Let me explain everything in detail."]  # sentence two never reached TTS
    last_bot = rig.of("bot")[-1]
    assert last_bot["interrupted"] is True and last_bot["text"] == "Let me explain everything in detail."
    assert rig.agent.history[-1] == {"role": "assistant", "content": "Let me explain everything in detail."}


def test_after_barge_in_the_next_turn_is_answered_normally() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM(["First answer that gets cut off.", "Second answer."]), FakeTTS(frames_per_sentence=100))
        rig.pacer.start()
        rig.event("EndOfTurn", "Question one.")
        await until(lambda: len(rig.frames) >= 3)
        rig.event("StartOfTurn", "Actually")
        rig.tts.frames = 2
        rig.event("EndOfTurn", "Actually, question two.")
        await settle(rig)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.tts.spoken == ["First answer that gets cut off.", "Second answer."]
    assert [m["role"] for m in rig.agent.history] == ["user", "assistant", "user", "assistant"]
    assert rig.agent.barge_ins == 1 and len(rig.of("turn")) == 2


def test_start_of_turn_when_bot_is_silent_is_not_a_barge_in() -> None:
    async def scenario() -> Rig:
        rig = Rig()
        rig.pacer.start()
        rig.event("StartOfTurn", "Hello")
        await asyncio.sleep(0.01)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.agent.barge_ins == 0 and rig.kinds() == []


def test_caller_keeps_talking_while_the_llm_request_is_in_flight() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM(["Never used.", "Combined answer."])
        llm.hold = asyncio.Event()  # the LLM has not answered yet
        rig = Rig(llm)
        rig.pacer.start()
        rig.event("EndOfTurn", "I want solar")
        await until(lambda: len(llm.requests) == 1)
        rig.event("StartOfTurn", "and")  # not audible yet -> quiet cancel, no barge-in
        llm.hold = None
        rig.event("EndOfTurn", "and also a battery.")
        await settle(rig)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.agent.barge_ins == 0
    assert rig.llm.cancelled == 1  # the in-flight request was cancelled
    assert rig.tts.spoken == ["Combined answer."]
    # both things the caller said reach the model as one message
    assert rig.llm.requests[1][-2] == {"role": "user", "content": "I want solar and also a battery."}


# ---------------------------------------------------------------------- eager end of turn
def test_eager_reply_is_held_back_until_end_of_turn_confirms() -> None:
    async def scenario() -> tuple[Rig, int]:
        rig = Rig()
        rig.pacer.start()
        rig.event("EagerEndOfTurn", "I want solar.")
        await until(lambda: len(rig.tts.spoken) == 2)  # LLM and TTS already ran
        await asyncio.sleep(0.05)
        while_unsure = len(rig.frames)
        rig.event("EndOfTurn", "I want solar.")
        await settle(rig)
        await rig.agent.close()
        return rig, while_unsure

    rig, while_unsure = run(scenario)
    assert while_unsure == 0  # caller heard nothing while we were only guessing
    assert len(rig.frames) == 4
    assert len(rig.llm.requests) == 1  # the head start was reused, not repeated
    assert rig.of("turn")[0]["eager"] is True
    assert sorted(rig.kinds()) == ["bot", "turn", "user"]


def test_repeated_eager_event_with_same_words_does_not_restart_the_llm() -> None:
    async def scenario() -> Rig:
        rig = Rig()
        rig.pacer.start()
        rig.event("EagerEndOfTurn", "I want solar.")
        await until(lambda: len(rig.llm.requests) == 1)
        rig.event("EagerEndOfTurn", "I want solar.")
        rig.event("EagerEndOfTurn", "I want solar.")
        rig.event("EndOfTurn", "I want solar.")
        await settle(rig)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert len(rig.llm.requests) == 1 and rig.llm.cancelled == 0


def test_eager_reply_is_discarded_when_the_caller_resumes() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM([reply("Wrong guess.", owns_home="no"), "Right answer."])
        rig = Rig(llm)
        rig.pacer.start()
        rig.event("EagerEndOfTurn", "I don't")
        await until(lambda: len(llm.requests) >= 1)
        await asyncio.sleep(0.03)
        rig.event("TurnResumed", "I don't rent")
        await asyncio.sleep(0.05)
        assert rig.frames == [] and rig.kinds() == []  # silently dropped
        assert rig.agent.slots["owns_home"] is None  # a discarded guess changes no state
        rig.event("EndOfTurn", "I don't rent, I own the house.")
        await settle(rig)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.agent.history[0] == {"role": "user", "content": "I don't rent, I own the house."}
    assert "Wrong guess." not in rig.bot_lines()
    assert rig.of("turn")[0]["eager"] is False and rig.agent.barge_ins == 0


def test_eager_guess_with_different_final_text_is_replaced() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM(["Guess.", "Real."]))
        rig.pacer.start()
        rig.event("EagerEndOfTurn", "Book Tuesday")
        await until(lambda: len(rig.llm.requests) == 1)
        rig.event("EndOfTurn", "Book Tuesday morning.")
        await settle(rig)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert len(rig.llm.requests) == 2
    assert rig.bot_lines() == ["Real."]


def test_eager_can_be_switched_off() -> None:
    async def scenario() -> Rig:
        rig = Rig(eager=False)
        rig.pacer.start()
        rig.event("EagerEndOfTurn", "Hi there.")
        await asyncio.sleep(0.02)
        await rig.agent.close()
        return rig

    assert run(scenario).llm.requests == []


# ---------------------------------------------------------------------- failures
def test_llm_failure_plays_apology_and_call_continues() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM([LLMError("Groq HTTP 429: rate limit"), "Recovered."]))
        rig.pacer.start()
        await rig.say("Hello?")
        assert len(rig.frames) == 2  # the cached apology
        await rig.say("Hello again.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    (error,) = rig.of("error")
    assert error["stage"] == "llm" and "429" in error["message"]
    apology = rig.of("bot")[0]
    assert apology["kind"] == "apology" and apology["text"] == APOLOGY_TEXT
    assert rig.bot_lines()[-1] == "Recovered." and rig.agent.errors == 1


def test_tts_failure_plays_pre_generated_apology() -> None:
    async def scenario() -> Rig:
        tts = FakeTTS()
        tts.fail = TTSError("Aura HTTP 500")
        rig = Rig(tts=tts)
        rig.pacer.start()
        await rig.say("Hello?")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.of("error")[0]["stage"] == "tts"
    assert rig.of("bot")[-1]["kind"] == "apology"
    assert len(rig.frames) == 2  # the apology came from the cache, not from Aura


def test_unexpected_exception_in_pipeline_does_not_escape() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM([RuntimeError("bug")]))
        rig.pacer.start()
        await rig.say("Hello?")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.of("error")[0]["stage"] == "agent"
    assert rig.of("bot")[0]["kind"] == "apology"


def test_reply_without_usable_say_is_an_error_not_silence() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM([{"slots": {}, "say": "   "}]))
        rig.pacer.start()
        await rig.say("Hello?")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.of("error")[0]["stage"] == "llm"
    assert rig.of("bot")[0]["kind"] == "apology"


def test_failed_eager_guess_is_silent_and_retried() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM([LLMError("Groq HTTP 503"), "Second try worked."]))
        rig.pacer.start()
        rig.event("EagerEndOfTurn", "Hi.")
        await until(lambda: rig.agent._job is not None and rig.agent._job.failed)
        assert rig.frames == []  # no apology for a guess nobody heard
        rig.event("EndOfTurn", "Hi.")
        await settle(rig)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines() == ["Second try worked."]


def test_greeting_failure_and_stt_error_are_logged_not_raised() -> None:
    async def scenario() -> Rig:
        rig = Rig(phrases=FakePhrases(missing=True))
        await rig.agent.start()
        rig.agent.on_stt_event(SttError("Deepgram Flux connection failed"), time.time())
        rig.agent.on_stt_event(FluxInfo("Connected"), time.time())
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert [e["stage"] for e in rig.of("error")] == ["tts", "stt"]
    assert rig.frames == []


# ---------------------------------------------------------------------- call length / close
def test_max_call_duration_says_goodbye_then_hangs_up() -> None:
    async def scenario() -> Rig:
        short = dataclasses.replace(PROFILE, max_call_seconds=0.05)  # type: ignore[arg-type]
        rig = Rig(FakeLLM(["A long answer."]), FakeTTS(frames_per_sentence=500), profile=short)
        await rig.agent.start()
        rig.event("EndOfTurn", "Tell me a story.")
        await until(lambda: bool(rig.hangups), timeout=3)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.hangups == ["max-duration"]
    assert rig.of("max_duration") == [{"type": "max_duration", "seconds": 0.05}]
    goodbye = rig.of("bot")[-1]
    assert goodbye["kind"] == "goodbye" and goodbye["text"] == goodbye_text(PROFILE)
    assert len(rig.frames) < 100  # the long answer was cut, not played to the end


def test_close_cancels_everything_in_flight() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM(["never"])
        llm.hold = asyncio.Event()
        rig = Rig(llm)
        rig.pacer.start()
        rig.event("EndOfTurn", "Hi.")
        await until(lambda: len(llm.requests) == 1)
        await rig.agent.close()  # caller hung up while the LLM was still working
        await asyncio.sleep(0.03)
        return rig

    rig = run(scenario)
    assert rig.llm.cancelled == 1 and rig.frames == [] and rig.hangups == []
