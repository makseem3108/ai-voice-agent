"""After the outcome is announced the model answers in plain sentences.

Replay of 2026-11-16: Groq rejected 4 of 16 JSON replies ("Failed to generate
JSON"), 3 of them on a caller's question after everything had been collected.
Nothing is extracted on those turns, so they no longer ask for JSON at all: no
schema, no slots, one request. Every turn before the outcome is unchanged, and
so is the turn after "we cannot help" (a caller may correct a misheard answer)."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import httpx
import pytest

from app.llm_groq import PATH_PLAIN, GroqLLM, LLMError, LLMUsage
from app.prompt_builder import FORMAT_REMINDER, PLAIN_REMINDER, build_directive, plain_reply
from app.slots import ANNOUNCE, ANSWER, COLLECT, CONFIRM, FINAL, OUTCOME_NOT_FIT, OUTCOME_QUALIFIED, Plan
from tests.helpers import NAME, OWNS_HOME, FakeLLM, Rig, reply, run


# ---------------------------------------------------------------- which turns are plain
@pytest.mark.parametrize(
    "plan, plain",
    [
        (Plan(ANSWER, outcome=OUTCOME_QUALIFIED), True),
        (Plan(ANSWER, outcome="qualified_no_callback"), True),
        (Plan(ANSWER, outcome=OUTCOME_NOT_FIT), False),  # a correction must still be heard
        (Plan(ANNOUNCE, outcome=OUTCOME_QUALIFIED), False),
        (Plan(COLLECT, target=NAME), False),
        (Plan(CONFIRM, reason=OWNS_HOME), False),
        (Plan(FINAL), False),
    ],
)
def test_only_turns_after_an_announced_outcome_are_plain(plan: Plan, plain: bool) -> None:
    assert plain_reply(plan) is plain
    directive = build_directive(plan, {})
    assert directive.endswith(PLAIN_REMINDER if plain else FORMAT_REMINDER)
    assert (FORMAT_REMINDER in directive) is (not plain)
    if plain:
        assert '"say"' not in directive and "slots" in PLAIN_REMINDER


# ---------------------------------------------------------------- the agent
def qualified_call(*after: Any) -> Rig:
    async def scenario() -> Rig:
        llm = FakeLLM([reply("Thanks.", owns_home="yes", monthly_bill="100 dollars", name="Maria"), reply("Thank you, Maria."), *after])
        rig = Rig(llm)
        rig.pacer.start()
        await rig.say("I own the house, the bill is 100 dollars, my name is Maria.")
        assert rig.agent.announced == "qualified"
        await rig.say("Is my appointment confirmed?")
        await rig.agent.close()
        return rig

    return run(scenario)


def test_question_after_the_outcome_is_asked_as_plain_text_without_a_schema() -> None:
    rig = qualified_call("Not yet. An advisor will call you to confirm.")
    assert rig.llm.plain == [False, False, True]  # collect, announce: JSON; the question: plain
    assert len(rig.llm.schemas) == 2  # no schema went out with the plain request
    assert rig.llm.directive().endswith(PLAIN_REMINDER)
    assert rig.bot_lines()[-1] == "Not yet. An advisor will call you to confirm."
    assert rig.of("turn")[-1]["llm_path"] == "plain"
    summary = rig.agent.summary()
    assert summary["llm_paths"]["plain"] == 1 and summary["llm_paths"]["json"] == 2
    assert summary["json_fallbacks"] == 0  # a plain turn is not a failed JSON turn


def test_plain_turn_never_changes_the_slots() -> None:
    # Even if a reply came back with slots in it, a plain turn does not read them.
    rig = qualified_call({"slots": {"name": "Someone Else", "owns_home": "no"}, "say": "We open at eight."})
    assert rig.agent.slots["name"] == "Maria" and rig.agent.slots["owns_home"] == "yes"
    assert rig.agent.announced == "qualified" and rig.bot_lines()[-1] == "We open at eight."


def test_after_not_a_fit_the_turn_is_still_json_so_a_correction_counts() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM([reply("Thanks.", owns_home="no"), reply("We only install on houses you own."), reply("Okay.")])
        rig = Rig(llm)
        rig.pacer.start()
        await rig.say("I rent.")
        assert rig.agent.announced == "not_fit"
        await rig.say("What are your opening hours?")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.llm.plain == [False, False, False]
    assert rig.llm.directive().endswith(FORMAT_REMINDER)


# ---------------------------------------------------------------- the Groq client
def groq(handler: Callable[[httpx.Request], httpx.Response]) -> GroqLLM:
    return GroqLLM(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "groq-test-key", "openai/gpt-oss-120b")


def text(content: Any) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 1500, "completion_tokens": 30}})


def ask(llm: GroqLLM) -> tuple[dict[str, Any], LLMUsage]:
    async def scenario() -> tuple[dict[str, Any], LLMUsage]:
        usage = LLMUsage()
        return await llm.reply_text([{"role": "user", "content": "hi"}], usage), usage

    return asyncio.run(scenario())


def test_plain_request_carries_no_schema_and_no_tools() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return text("  لسا مش مأكّد. العيادة بتتصل فيك.  ")

    result, usage = ask(groq(handler))
    assert result == {"say": "لسا مش مأكّد. العيادة بتتصل فيك.", "path": PATH_PLAIN}
    assert len(bodies) == 1 and usage.requests == 1 and usage.prompt_tokens == 1500
    assert "response_format" not in bodies[0] and "tools" not in bodies[0] and "tool_choice" not in bodies[0]
    assert bodies[0]["stream"] is False


def test_json_written_out_of_habit_is_unwrapped() -> None:
    wrapped = json.dumps({"slots": {"name": None}, "say": "We open at eight."})
    assert ask(groq(lambda request: text(wrapped)))[0] == {"say": "We open at eight.", "path": PATH_PLAIN}
    only_say = json.dumps({"say": "We open at eight."})  # told "no slots", the model may leave them out
    assert ask(groq(lambda request: text(only_say)))[0] == {"say": "We open at eight.", "path": PATH_PLAIN}


@pytest.mark.parametrize("content", ["", "   ", None, '{"slots": {', '{"note": "no say here"}'])
def test_empty_or_broken_plain_reply_is_an_error(content: Any) -> None:
    with pytest.raises(LLMError):
        ask(groq(lambda request: text(content)))


def test_plain_request_that_is_refused_is_an_error_without_a_second_request() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500, json={"error": {"message": "internal"}})

    with pytest.raises(LLMError, match="HTTP 500"):
        ask(groq(handler))
    assert len(calls) == 1
