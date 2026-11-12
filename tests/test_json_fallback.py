"""Safety net: the model answered in plain sentences although JSON was asked for.

The exact situation of the real call on 2026-11-12: "tell me more about your
service" -> Groq HTTP 400 "Failed to generate JSON" three times -> apology."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Optional

import httpx
import pytest

from app.call_log import format_record
from app.llm_groq import GroqLLM, LLMError, LLMUsage
from app.text_utils import clean_fallback_text
from tests.helpers import PROFILE, FakeLLM, Rig, reply, run, until

REAL = "We offer free site visits and quotes for rooftop solar panels on homes. Do you own the house, or do you rent?"
SCHEMA: dict[str, Any] = {"type": "object"}


# ====================================================================== what counts as clean
@pytest.mark.parametrize(
    "text",
    [
        REAL,
        "We install rooftop solar panels (and batteries) on owner‑occupied homes, with a 25‑year performance warranty.",
        "I’m sorry, I don’t have that information.  Do you own the house,\n or do you rent?",
    ],
)
def test_clean_english_text_is_accepted(text: str) -> None:
    assert clean_fallback_text(text, "en") == " ".join(text.split())


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "?!",
        '{"slots": {"owns_home": null}, "say": "Do you own the house?"',  # broken JSON
        'say: "Do you own the house?"',
        "slots = none. Do you own the house?",
        "Sure! **We offer** free site visits.",  # markdown
        "# Our service\nWe offer free site visits.",
        "- free site visits\n- quotes",
        "We offer `free` site visits.",
        "Fill owns_home first.",  # internal names leaking
        "We offer free site visits <br> and quotes.",
        "x" * 300,  # too long to be a spoken reply
        "שלום, אנחנו מתקינים פאנלים סולאריים.",  # wrong language for an English profile
        "مرحبا، نحن نركب ألواح شمسية.",
        None,
        42,
    ],
)
def test_unclean_text_is_refused(text: Any) -> None:
    assert clean_fallback_text(text, "en") is None


def test_language_is_checked_by_alphabet() -> None:
    hebrew = "אנחנו מתקינים פאנלים סולאריים על בתים פרטיים."
    arabic = "نحن نركب ألواح شمسية على البيوت."
    assert clean_fallback_text(hebrew, "he") == hebrew and clean_fallback_text(arabic, "ar-JO") == arabic
    assert clean_fallback_text(hebrew, "ar") is None and clean_fallback_text(arabic, "he") is None
    assert clean_fallback_text(REAL, "he") is None
    assert clean_fallback_text("יש לנו אחריות של 25 שנה על הפאנלים, כולל WiFi.", "he") is not None  # a foreign word is fine


# ====================================================================== the Groq client
def groq(handler: Callable[[httpx.Request], httpx.Response], language: Optional[str] = "en") -> GroqLLM:
    llm = GroqLLM(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "groq-test-key", "openai/gpt-oss-120b")
    llm.fallback_language = language
    return llm


def rejected(generated: str) -> httpx.Response:
    return httpx.Response(
        400,
        json={"error": {"message": "Failed to generate JSON. Please adjust your prompt.", "code": "json_validate_failed", "failed_generation": generated}},
    )


def ok(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


def ask(llm: GroqLLM, **kwargs: Any) -> tuple[dict[str, Any], LLMUsage]:
    async def scenario() -> tuple[dict[str, Any], LLMUsage]:
        usage = LLMUsage()
        return await llm.reply_json([{"role": "user", "content": "hi"}], SCHEMA, usage, **kwargs), usage

    return asyncio.run(scenario())


def mode(request: httpx.Request) -> str:
    body = json.loads(request.content)
    return "function_call" if "tools" in body else body["response_format"]["type"]


def test_clean_rejected_text_is_used_only_after_the_function_call_also_failed() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(mode(request))
        return rejected(REAL)

    result, usage = ask(groq(handler))
    assert result == {"say": REAL, "path": "text_fallback"}  # no "slots": nothing is changed
    assert calls == ["json_schema", "function_call"] and usage.requests == 2


def test_the_first_clean_text_is_kept_if_the_second_attempt_is_garbage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return rejected(REAL) if mode(request) == "json_schema" else rejected('{"slots": {')

    assert ask(groq(handler))[0] == {"say": REAL, "path": "text_fallback"}


def test_unclean_text_both_times_still_ends_in_an_error() -> None:
    with pytest.raises(LLMError, match="no usable reply after retry"):
        ask(groq(lambda request: rejected("**Sure!** {")))


def test_fallback_can_be_switched_off_per_request_and_is_off_without_a_language() -> None:
    with pytest.raises(LLMError):
        ask(groq(lambda request: rejected(REAL)), text_fallback=False)
    with pytest.raises(LLMError):
        ask(groq(lambda request: rejected(REAL), language=None))
    with pytest.raises(LLMError):
        ask(groq(lambda request: rejected(REAL), language="he"))  # English text on a Hebrew call


def test_plain_text_in_a_200_reply_is_also_a_candidate() -> None:
    """The function call can come back as ordinary content instead of a call."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": REAL}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    assert ask(groq(handler))[0] == {"say": REAL, "path": "text_fallback"}


# ====================================================================== the agent
def test_real_call_5_question_gets_the_plain_text_answer_and_slots_stay() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM([reply("Great. Do you own the house, or do you rent?"), {"say": REAL, "path": "text_fallback"}])
        rig = Rig(llm)
        rig.pacer.start()
        await rig.say("Hi.")
        before = dict(rig.agent.slots)
        await rig.say("Alright. Tell me more about your service.")
        assert rig.agent.slots == before
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines()[1] == REAL and rig.of("error") == [] and rig.agent.errors == 0
    assert "Sorry, I'm having a little trouble" not in " ".join(rig.bot_lines())
    (record,) = rig.of("llm_json_fallback")
    assert record == {"type": "llm_json_fallback", "turn": 2, "text": REAL}
    summary = rig.agent.summary()
    assert summary["json_fallbacks"] == 1 and summary["llm_paths"] == {"json": 1, "function_call": 0, "text_fallback": 1}
    assert [t["llm_path"] for t in rig.of("turn")] == ["json", "text_fallback"]  # which path answered each turn
    line = format_record(record)
    assert line is not None and line.startswith("[llm_json_fallback]")


def test_function_call_turn_is_logged_and_keeps_the_slots() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM([{**reply("Thanks. How much is your bill?", owns_home="yes"), "path": "function_call"}]))
        rig.pacer.start()
        await rig.say("I own it.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.agent.slots["owns_home"] == "yes" and rig.of("llm_json_fallback") == []
    assert rig.of("turn")[0]["llm_path"] == "function_call" and rig.of("state")[-1]["llm_path"] == "function_call"
    assert rig.agent.summary()["llm_paths"] == {"json": 0, "function_call": 1, "text_fallback": 0}
    assert "llm_path=function_call" in (format_record(rig.of("turn")[0]) or "")


# ====================================================================== after a fallback: settle the deciding slot first
MOVED_ON = "Got it, thanks. Roughly how much is your monthly electricity bill?"  # what the model really said in the replay


def test_apartment_answer_lost_in_a_fallback_is_recovered_on_the_next_turn() -> None:
    """Replay 2026-11-12: "No. I live in an apartment." -> plain-text fallback that moved on to the bill."""

    async def scenario() -> Rig:
        llm = FakeLLM(
            [
                {"say": MOVED_ON, "path": "text_fallback"},
                reply("Thanks.", owns_home="no", monthly_bill="500 shekels"),  # now read from the conversation
                reply("We only install on houses you own, but your landlord can contact us."),
            ]
        )
        rig = Rig(llm)
        rig.pacer.start()
        await rig.say("No. I live in an apartment.")
        assert rig.agent.slots["owns_home"] is None and rig.of("state")[-1]["recheck"] == "owns_home"
        await rig.say("About five hundred shekels.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    recheck = rig.llm.directive(1)
    assert "Still needed: owns_home" in recheck and "may not have been recorded" in recheck and "Ask about nothing else" in recheck
    assert "monthly electricity bill" not in recheck  # it must not be told to move on
    assert rig.agent.slots["owns_home"] == "no" and rig.agent.replans == 1
    assert rig.bot_lines()[-1].startswith("We only install on houses you own")  # the not-a-fit announcement, not the next question
    assert rig.of("state")[-1]["outcome"] == "not_fit" and rig.of("state")[-1]["recheck"] is None
    assert "(will re-check owns_home)" in (format_record(rig.of("state")[0]) or "")


def test_deciding_slot_is_asked_again_if_it_still_is_not_known() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM(
            [
                {"say": MOVED_ON, "path": "text_fallback"},
                reply("Sorry, one thing first: do you own the house, or do you rent?", monthly_bill="500 shekels"),
                reply("Thanks. What name should the advisor ask for?", owns_home="yes"),
            ]
        )
        rig = Rig(llm)
        rig.pacer.start()
        await rig.say("Mm, it's complicated.")
        await rig.say("About five hundred shekels.")
        assert rig.agent.replans == 0 and rig.agent.slots["owns_home"] is None  # still open: the re-ask is spoken as is
        assert rig.of("state")[-1]["next_slot"] == "owns_home" and rig.of("state")[-1]["recheck"] is None
        await rig.say("I own it.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines()[1].endswith("do you own the house, or do you rent?")
    assert "may not have been recorded" not in rig.llm.directive(2)  # only the one turn after the fallback
    assert rig.agent.slots["owns_home"] == "yes" and rig.agent.slots["monthly_bill"] == "500 shekels"


def test_no_recheck_for_a_slot_that_cannot_change_the_outcome() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM([{"say": "Sure. What name should the advisor ask for?", "path": "text_fallback"}, reply("Thanks, Dana.", name="Dana")])
        rig = Rig(llm)
        rig.agent.slots.update({"owns_home": "yes"})  # next slot is the bill: any value qualifies
        rig.pacer.start()
        await rig.say("What was the question?")
        assert rig.of("state")[-1]["recheck"] is None
        await rig.say("Dana.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert "may not have been recorded" not in rig.llm.directive(1)


def test_goodbye_extraction_never_accepts_plain_text() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM([reply("Okay.", owns_home="yes")]))
        rig.pacer.start()
        await rig.say("Yes I own it. Goodbye.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.llm.text_fallback == [False]  # only the slots matter there


def test_normal_turns_allow_the_fallback() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM([reply("Do you own the house?")]))
        rig.pacer.start()
        await rig.say("Hi.")
        await rig.agent.close()
        return rig

    assert run(scenario).llm.text_fallback == [True]
