"""Slot filling: the code owns outcome, next question, "anything else?" and the
closing line. Includes the exact situations from the three real test calls."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import httpx
import pytest

from app.call_log import format_record
from app.llm_groq import GroqLLM, LLMError, LLMUsage
from app.prompt_builder import ANYTHING_ELSE_QUESTION, build_directive, incomplete_closing
from app.slots import (
    ANNOUNCE,
    ANSWER,
    COLLECT,
    SlotDef,
    compute_outcome,
    empty_slots,
    make_plan,
    merge_slots,
    missing_slots,
    normalise_value,
    parse_slots,
    plans_consistent,
    reply_schema,
)
from tests.helpers import BILL, FRAME, NAME, OWNS_HOME, PROFILE, FakeLLM, FakeTTS, Rig, reply, run, settle, until

DEFS = PROFILE.slots
QUALIFIED = PROFILE.closing_qualified
NOT_FIT = PROFILE.closing_not_fit


def filled(**values: Any) -> dict[str, Any]:
    return {**empty_slots(DEFS), **values}


# ====================================================================== slot definitions (profile validation)
def slot(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"name": "owns_home", "type": "yes_no", "question": "Do you own the house?"}
    base.update(overrides)
    return base


def test_valid_slots_parse() -> None:
    slots, problems = parse_slots(
        [
            slot(disqualifying=["no"], not_fit_reason="we only  install on owned houses", description="yes if owner"),
            {"name": "plan", "type": "choice", "values": ["Basic", "Pro"], "question": "Which plan?"},
            {"name": "bill", "type": "number", "question": "How much is your bill?"},
            {"name": "hint", "type": "text", "required": False},
        ]
    )
    assert problems == []
    owns, plan, bill, hint = slots
    assert owns.allowed == ("yes", "no") and owns.disqualifying == ("no",)
    assert owns.not_fit_reason == "we only install on owned houses"  # whitespace normalised
    assert plan.allowed == ("basic", "pro") and bill.allowed is None
    assert hint.required is False and hint.question is None


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (None, "missing required field 'slots'"),
        ([], "'slots' must be a non-empty list"),
        ("owns_home", "'slots' must be a non-empty list"),
        ([slot(name="Owns Home")], "'name' must be lowercase letters"),
        ([slot(), slot()], "slot 'owns_home' is defined twice"),
        ([slot(type="boolean")], "'type' must be one of yes_no, choice, text, number"),
        ([slot(question=None)], "a required slot needs a 'question'"),
        ([slot(type="choice")], "a 'choice' slot needs 'values' with at least two"),
        ([slot(values=["a", "b"])], "'values' is only allowed for type 'choice'"),
        ([slot(disqualifying=["maybe"], not_fit_reason="x")], "disqualifying value\\(s\\) maybe are not allowed"),
        ([slot(disqualifying=["yes", "no"], not_fit_reason="x")], "nobody could ever qualify"),
        ([slot(disqualifying=["no"])], "'not_fit_reason' is required when 'disqualifying' is set"),
        ([slot(type="text", disqualifying=["x"], not_fit_reason="x")], "'disqualifying' needs a 'yes_no' or 'choice' slot"),
        ([slot(colour="red")], "unknown field\\(s\\): colour"),
        ([slot(required="yes")], "'required' must be true or false"),
        ([slot(required=False)], "at least one slot must be required"),
        ([slot(name=f"s{i}") for i in range(13)], "keep it to 12 or fewer"),
        ([["not", "a", "mapping"]], "slot #1 must be a mapping"),
    ],
)
def test_invalid_slot_definitions_are_reported(data: Any, message: str) -> None:
    import re

    _, problems = parse_slots(data)
    assert any(re.search(message, p) for p in problems), problems


# ====================================================================== values
def test_normalise_values_by_type() -> None:
    assert normalise_value(OWNS_HOME, "Yes") == "yes"
    assert normalise_value(OWNS_HOME, True) == "yes" and normalise_value(OWNS_HOME, False) == "no"
    assert normalise_value(OWNS_HOME, "maybe") is None  # not an allowed value
    for unknown in (None, "", "unknown", "N/A", "null", "not sure", [], {}):
        assert normalise_value(OWNS_HOME, unknown) is None
        assert normalise_value(NAME, unknown) is None
    assert normalise_value(NAME, "  Maria   Lopez ") == "Maria Lopez"
    assert len(str(normalise_value(NAME, "x" * 500))) == 120
    number = SlotDef(name="bill", type="number", question="?")
    assert normalise_value(number, 150) == 150 and normalise_value(number, 99.5) == 99.5
    assert normalise_value(number, "about 150 dollars") == 150
    assert normalise_value(number, "12,5") == 12.5
    assert normalise_value(number, "a lot") is None and normalise_value(number, True) is None
    choice = SlotDef(name="plan", type="choice", values=("basic", "pro"), question="?")
    assert normalise_value(choice, "PRO") == "pro" and normalise_value(choice, "gold") is None


def test_merge_keeps_known_values_accepts_corrections_and_rejects_inventions() -> None:
    start = filled(owns_home="yes")
    merged, rejected = merge_slots(DEFS, start, {"owns_home": None, "monthly_bill": "150 dollars", "name": "unknown"})
    assert merged == filled(owns_home="yes", monthly_bill="150 dollars") and rejected == []  # null never erases
    merged, rejected = merge_slots(DEFS, merged, {"owns_home": "no"})  # caller corrected themselves
    assert merged["owns_home"] == "no"
    merged, rejected = merge_slots(DEFS, start, {"owns_home": "perhaps", "favourite_colour": "blue"})
    assert merged == start
    assert rejected == ["owns_home: 'perhaps' is not a valid value", "favourite_colour: unknown slot"]
    assert start == filled(owns_home="yes")  # input not modified
    assert merge_slots(DEFS, start, None) == (start, [])
    assert merge_slots(DEFS, start, "yes")[1] == ["slots is not an object"]


# ====================================================================== outcome and plan
def test_outcome_is_computed_from_slots_only() -> None:
    assert compute_outcome(DEFS, empty_slots(DEFS)) is None
    assert compute_outcome(DEFS, filled(owns_home="yes")) is None
    assert compute_outcome(DEFS, filled(owns_home="yes", monthly_bill="100", name="Al")) == "qualified"
    # a disqualifying value wins, whatever else is known or missing
    assert compute_outcome(DEFS, filled(owns_home="no")) == "not_fit"
    assert compute_outcome(DEFS, filled(owns_home="no", monthly_bill="100", name="Al")) == "not_fit"


def test_optional_inferred_slot_can_disqualify_but_is_never_asked() -> None:
    inferred = SlotDef(name="offered", type="yes_no", required=False, disqualifying=("no",), not_fit_reason="not offered")
    defs = (inferred, NAME)
    assert [s.name for s in missing_slots(defs, empty_slots(defs))] == ["name"]
    assert compute_outcome(defs, {"offered": None, "name": "Al"}) == "qualified"
    assert compute_outcome(defs, {"offered": "no", "name": None}) == "not_fit"


def test_plan_picks_the_next_missing_slot_in_profile_order() -> None:
    plan = make_plan(DEFS, empty_slots(DEFS), None)
    assert (plan.kind, plan.target, plan.next) == (COLLECT, OWNS_HOME, BILL)
    plan = make_plan(DEFS, filled(owns_home="yes"), None)
    assert (plan.target, plan.next) == (BILL, NAME)
    plan = make_plan(DEFS, filled(owns_home="yes", monthly_bill="100"), None)
    assert (plan.target, plan.next) == (NAME, None)
    # out of order: the caller gave their name first
    assert make_plan(DEFS, filled(name="Al"), None).target == OWNS_HOME


def test_plan_announces_an_outcome_once_then_only_answers() -> None:
    qualified = filled(owns_home="yes", monthly_bill="100", name="Al")
    plan = make_plan(DEFS, qualified, None)
    assert (plan.kind, plan.outcome) == (ANNOUNCE, "qualified")
    assert make_plan(DEFS, qualified, "qualified").kind == ANSWER
    not_fit = make_plan(DEFS, filled(owns_home="no"), None)
    assert (not_fit.kind, not_fit.outcome, not_fit.reason) == (ANNOUNCE, "not_fit", OWNS_HOME)
    assert make_plan(DEFS, filled(owns_home="no"), "not_fit").kind == ANSWER
    # the caller corrected themselves after a not-a-fit announcement: back to collecting
    assert make_plan(DEFS, filled(owns_home="yes"), "not_fit").kind == COLLECT
    # ... and a later qualified outcome is announced even though not_fit was announced before
    assert make_plan(DEFS, qualified, "not_fit").kind == ANNOUNCE


def test_plan_consistency() -> None:
    start = make_plan(DEFS, empty_slots(DEFS), None)
    assert plans_consistent(start, start)  # nothing learned: ask the same slot again
    assert plans_consistent(start, make_plan(DEFS, filled(owns_home="yes"), None))  # answered: ask the next one
    assert not plans_consistent(start, make_plan(DEFS, filled(owns_home="yes", monthly_bill="1"), None))  # two at once
    assert not plans_consistent(start, make_plan(DEFS, filled(owns_home="no"), None))  # outcome decided
    answer = make_plan(DEFS, filled(owns_home="no"), "not_fit")
    assert plans_consistent(answer, answer)
    assert not plans_consistent(answer, make_plan(DEFS, filled(owns_home="yes"), "not_fit"))


def test_reply_schema_is_strict_and_puts_slots_before_say() -> None:
    schema = reply_schema(DEFS)
    assert list(schema["properties"]) == ["slots", "say"] and schema["required"] == ["slots", "say"]
    assert schema["additionalProperties"] is False
    slots = schema["properties"]["slots"]
    assert slots["required"] == ["owns_home", "monthly_bill", "name"] and slots["additionalProperties"] is False
    assert slots["properties"]["owns_home"] == {"type": ["string", "null"], "enum": ["yes", "no", None]}
    assert slots["properties"]["name"] == {"type": ["string", "null"]}
    number = reply_schema((SlotDef(name="bill", type="number", question="?"),))
    assert number["properties"]["slots"]["properties"]["bill"] == {"type": ["number", "null"]}
    json.dumps(schema)  # must be serialisable as-is


# ====================================================================== directives
def test_collect_directive_names_the_slot_and_both_questions() -> None:
    text = build_directive(make_plan(DEFS, empty_slots(DEFS), None), empty_slots(DEFS))
    assert text.startswith("INSTRUCTION")
    assert "Known so far: nothing yet" in text
    assert "Still needed: owns_home" in text
    assert OWNS_HOME.question in text and BILL.question in text
    known = filled(owns_home="yes")
    text = build_directive(make_plan(DEFS, known, None), known)
    assert 'Known so far: {"owns_home": "yes"}' in text and "Still needed: monthly_bill" in text


def test_replanned_collect_directive_asks_exactly_one_question() -> None:
    known = filled(owns_home="yes", monthly_bill="100")
    text = build_directive(make_plan(DEFS, known, None), known, replanned=True)
    assert NAME.question in text and "If it did give it" not in text


def test_announce_directives_forbid_questions_and_goodbyes() -> None:
    not_fit = filled(owns_home="no")
    text = build_directive(make_plan(DEFS, not_fit, None), not_fit)
    assert "We cannot help this caller because we only install on houses the caller owns" in text
    assert "one thing from the knowledge that could still be useful" in text
    assert "Do not ask a question. Do not say goodbye." in text
    qualified = filled(owns_home="yes", monthly_bill="100", name="Al")
    text = build_directive(make_plan(DEFS, qualified, None), qualified)
    assert "Everything needed has been collected" in text and "Do not say goodbye." in text
    text = build_directive(make_plan(DEFS, qualified, "qualified"), qualified)
    assert "already been asked whether there is anything else" in text and "Do not ask a question" in text


# ====================================================================== flow: collecting
def test_code_tells_the_model_which_slot_to_ask_and_tracks_state_per_turn() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM(
            [
                reply("Great, thanks. Roughly how much is your electricity bill?", owns_home="yes"),
                reply("Got it. And what name should the advisor ask for?", monthly_bill="about 150 dollars"),
            ]
        )
        rig = Rig(llm)
        rig.pacer.start()
        await rig.say("I own my house.")
        await rig.say("About 150 dollars.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert len(rig.llm.requests) == 2  # one request per turn: the pre-planned next question fitted
    assert "Still needed: owns_home" in rig.llm.directive(0) and BILL.question in rig.llm.directive(0)
    assert "Still needed: monthly_bill" in rig.llm.directive(1) and NAME.question in rig.llm.directive(1)
    assert '"owns_home": "yes"' in rig.llm.directive(1)  # the model is told what is already known
    assert rig.agent.slots == {"owns_home": "yes", "monthly_bill": "about 150 dollars", "name": None}
    first, second = rig.of("state")
    assert first["slots"] == {"owns_home": "yes", "monthly_bill": None, "name": None}
    assert (first["outcome"], first["phase"], first["next_slot"]) == (None, "collect", "monthly_bill")
    assert (second["turn"], second["next_slot"], second["replanned"]) == (2, "name", False)
    assert rig.llm.schemas[0]["properties"]["slots"]["required"] == ["owns_home", "monthly_bill", "name"]
    line = format_record(first) or ""
    assert line.startswith("[state] outcome=undecided phase=collect next=monthly_bill") and "owns_home=yes" in line


def test_two_answers_at_once_trigger_one_replan_with_the_right_question() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM(
            [
                reply("Thanks. How much is your bill?", owns_home="yes", monthly_bill="100 dollars"),  # stale question
                reply("Perfect. What name should the advisor ask for?"),
            ]
        )
        rig = Rig(llm)
        rig.pacer.start()
        await rig.say("I own my house and my bill is 100 dollars.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert len(rig.llm.requests) == 2 and rig.agent.replans == 1
    assert "Still needed: name" in rig.llm.directive(1) and "If it did give it" not in rig.llm.directive(1)
    assert rig.bot_lines() == ["Perfect. What name should the advisor ask for?"]  # the stale reply was never spoken
    assert "How much is your bill?" not in " ".join(rig.tts.spoken)
    assert rig.of("state")[0]["replanned"] is True
    (raw,) = rig.of("llm_raw")
    assert raw["replanned"] is True and "How much is your bill?" in raw["text"]  # both replies kept for diagnosis


def test_values_the_model_invents_are_rejected_and_logged() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM([reply("Okay. Do you own the house?", owns_home="perhaps", shoe_size="42")]))
        rig.pacer.start()
        await rig.say("Hmm, it's complicated.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.agent.slots == empty_slots(DEFS)
    assert rig.of("state")[0]["rejected"] == ["owns_home: 'perhaps' is not a valid value", "shoe_size: unknown slot"]


def test_bare_no_while_collecting_is_an_answer_not_a_goodbye() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM([reply("Thanks.", owns_home="no"), reply("We only install on houses you own. A landlord can contact us.")])
        rig = Rig(llm)
        rig.pacer.start()
        await rig.say("No.")  # answer to "Do you own the house?"
        await asyncio.sleep(0.15)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.hangups == [] and rig.of("ending") == []
    assert len(rig.llm.requests) == 2  # it went to the model like any other answer
    assert rig.agent.slots["owns_home"] == "no"


def test_goodbye_before_any_outcome_gets_the_neutral_closing() -> None:
    async def scenario() -> Rig:
        rig = Rig()
        rig.pacer.start()
        await rig.say("Sorry, wrong number. Bye!")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    # the goodbye sentence is read for slot values (nothing in it here), never spoken to
    assert len(rig.llm.requests) == 1 and "The caller is ending the call" in rig.llm.directive(0)
    assert rig.tts.spoken == []
    assert rig.bot_lines() == [incomplete_closing(PROFILE)] == ["Thank you for calling Acme Solar. Goodbye!"]
    assert rig.hangups == ["caller-goodbye"] and rig.agent.summary()["outcome"] == "incomplete"


# ====================================================================== flow: qualified
def test_qualified_flow_announce_ask_once_then_fixed_closing() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM(
            [
                reply("Thanks.", owns_home="yes", monthly_bill="100 dollars", name="Maria"),  # written under the old plan
                reply("Thank you, Maria. An advisor will call you on a weekday."),  # announce
            ]
        )
        rig = Rig(llm, end_silence_s=0.1)
        rig.pacer.start()
        await rig.say("I own the house, the bill is 100 dollars, my name is Maria.")
        assert rig.hangups == [] and rig.of("ending") == []
        requests_before_close = len(llm.requests)
        await rig.say("No, that's all, thank you.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        rig.requests_before_close = requests_before_close  # type: ignore[attr-defined]
        return rig

    rig = run(scenario)
    assert "Everything needed has been collected" in rig.llm.directive(1)
    assert rig.bot_lines() == [
        f"Thank you, Maria. An advisor will call you on a weekday. {ANYTHING_ELSE_QUESTION}",
        QUALIFIED,
    ]
    assert rig.of("bot")[-1]["kind"] == "closing"
    assert len(rig.llm.requests) == rig.requests_before_close == 2  # type: ignore[attr-defined]  # no LLM for the closing
    assert QUALIFIED not in rig.tts.spoken and QUALIFIED in rig.phrases.served  # a fixed, pre-generated phrase
    assert rig.hangups == ["goal-complete"]
    summary = rig.agent.summary()
    assert summary["outcome"] == "qualified" and summary["slots"]["name"] == "Maria"
    assert rig.of("ending") == [{"type": "ending", "outcome": "qualified", "silence_s": 0.1, "max_s": 1.0}]
    assert rig.agent.announced == "qualified"


# ====================================================================== the three real calls
def test_real_call_3_apartment_gets_the_not_fit_closing_only() -> None:
    """Call c52dc050: the caller lives in an apartment; the bot closed with the
    QUALIFIED line ("an advisor will call you back") and asked "anything else?"
    three replies running. Same words from the caller, new engine."""

    async def scenario() -> Rig:
        llm = FakeLLM(
            [
                reply("I'm doing well, thank you! Do you own the house, or do you rent?"),
                reply("Roughly how much is your bill?", owns_home="no"),  # model still follows the old plan
                reply("We only install solar on houses you own, not apartments. A landlord can contact us for the building."),
                # from here on the model misbehaves exactly like in the real call:
                reply("Understood. Is there anything else I can assist you with?"),
                reply("I'm sorry, I don't have that information. Is there anything else I can help you with?"),
            ]
        )
        rig = Rig(llm, end_silence_s=0.1)
        await rig.agent.start()
        await settle(rig)
        await rig.say("Hi. How are you today?")
        await rig.say("I am live in apartment.")
        await rig.say("So rental house doesn't work. Only with the house.")
        await rig.say("How are the waiters today?")
        assert rig.hangups == [] and rig.of("ending") == []
        await rig.say("No. Okay. Thank you. Bye bye.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    everything_said = " ".join(rig.bot_lines())
    # 1. the right closing, and never the wrong one
    assert rig.bot_lines()[-1] == NOT_FIT
    assert QUALIFIED not in everything_said and "advisor will call you back" not in everything_said
    assert rig.hangups == ["not-a-fit"] and rig.agent.summary()["outcome"] == "not_fit"
    # 2. "anything else?" exactly once in the whole call, asked by the server
    assert everything_said.lower().count("anything else") == 1
    assert rig.bot_lines()[2].endswith(ANYTHING_ELSE_QUESTION)
    assert rig.bot_lines()[3] == "Understood."  # the model's own "anything else?" was dropped
    assert rig.bot_lines()[4] == "I'm sorry, I don't have that information."
    # 3. the outcome came from the slot, and the model was told why
    assert rig.agent.slots["owns_home"] == "no"
    assert "We cannot help this caller because we only install on houses the caller owns" in rig.llm.directive(2)
    assert "Roughly how much is your bill?" not in " ".join(rig.tts.spoken)  # stale question never spoken
    states = rig.of("state")
    assert [s["outcome"] for s in states] == [None, "not_fit", "not_fit", "not_fit", "not_fit"]
    assert [s["phase"] for s in states] == ["collect", "answer", "answer", "answer", "answer"]


def test_real_call_2_statement_is_not_a_decline() -> None:
    """Call 01069001: after "anything else?" the caller made a STATEMENT and the
    bot said goodbye and hung up."""

    async def scenario() -> Rig:
        llm = FakeLLM(
            [
                reply("Thanks.", owns_home="no"),
                reply("We only install on houses you own. A landlord can contact us."),
                reply("We install rooftop solar panels and home batteries for houses."),
                reply("That's right, only for houses the caller owns."),
            ]
        )
        rig = Rig(llm, end_silence_s=0.05)
        rig.pacer.start()
        await rig.say("I live in apartment.")
        await rig.say("Yes. What you are providing, what services?")
        await rig.say("only for own houses, not apartment or rental houses.")
        await asyncio.sleep(0.25)  # five times the silence window
        assert rig.hangups == [] and rig.of("ending") == []  # <- used to hang up here
        assert rig.bot_lines()[-1] == "That's right, only for houses the caller owns."
        await rig.say("No, thank you. Bye.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.hangups == ["not-a-fit"]
    assert rig.bot_lines()[-1] == NOT_FIT
    assert " ".join(rig.bot_lines()).lower().count("anything else") == 1


def test_real_call_1_off_topic_question_never_ends_the_call() -> None:
    """Call cb7a80da: "How is the weather today?" made the bot say goodbye and hang up."""

    async def scenario() -> Rig:
        llm = FakeLLM(
            [
                # even if the model tries to wrap up, nothing it says can end the call
                reply("I'm sorry, I don't have that information. Thanks, that's everything I need. Have a great day!"),
                reply("Great. Roughly how much is your electricity bill?", owns_home="yes"),
            ]
        )
        rig = Rig(llm, end_silence_s=0.05)
        rig.pacer.start()
        await rig.say("How is the weather today?")
        await asyncio.sleep(0.25)
        assert rig.hangups == [] and rig.of("ending") == []
        assert rig.agent.summary()["outcome"] is None
        await rig.say("Okay. I own my house.")  # the conversation simply goes on
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.hangups == []
    assert rig.agent.slots["owns_home"] == "yes"
    assert rig.of("state")[-1]["next_slot"] == "monthly_bill"
    assert QUALIFIED not in rig.bot_lines() and NOT_FIT not in rig.bot_lines()


# ====================================================================== flow: edge cases
def test_caller_corrects_themselves_after_not_fit_and_can_still_qualify() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM(
            [
                reply("Thanks.", owns_home="no"),
                reply("We only install on houses you own."),
                reply("Oh, good.", owns_home="yes"),  # written as an "answer", but the outcome changed
                reply("Great, then we can help. Roughly how much is your electricity bill?"),
            ]
        )
        rig = Rig(llm)
        rig.pacer.start()
        await rig.say("I rent.")
        assert rig.agent.announced == "not_fit"
        await rig.say("Sorry, I meant I own the house, I rent out a room.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.agent.slots["owns_home"] == "yes"
    assert "Still needed: monthly_bill" in rig.llm.directive(3)
    last = rig.of("state")[-1]
    assert (last["outcome"], last["phase"], last["next_slot"]) == (None, "collect", "monthly_bill")
    assert rig.hangups == []


def test_decline_right_after_the_announcement_closes_without_another_llm_call() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM([reply("Thanks.", owns_home="no"), reply("We only install on houses you own.")]), end_silence_s=0.05)
        rig.pacer.start()
        await rig.say("I rent an apartment.")
        await rig.say("No.")  # now, after "anything else?", a bare no IS a decline
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert len(rig.llm.requests) == 2 and rig.bot_lines()[-1] == NOT_FIT
    turn = rig.of("turn")[-1]
    assert turn["llm_first_token_ms"] is None and turn["total_ms"] is not None  # closing needs no LLM


def test_caller_speaking_during_or_after_the_closing_cancels_the_hangup() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM([reply("Thanks.", owns_home="no"), reply("We only install on houses you own."), reply("The site visit is free.")])
        rig = Rig(llm, end_silence_s=0.2)
        rig.pacer.start()
        await rig.say("I rent.")
        await rig.say("No thanks.")  # closing line plays, hang-up is pending
        assert rig.of("ending") != [] and rig.hangups == []
        await rig.say("Wait, one more thing, is the visit free?")  # inside the silence window
        await asyncio.sleep(0.4)
        assert rig.hangups == []  # the call was not cut
        assert rig.agent.summary()["outcome"] is None
        await rig.say("Okay, bye.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.kinds().count("hangup_cancelled") == 1
    assert rig.bot_lines()[-3:] == [NOT_FIT, "The site visit is free.", NOT_FIT]
    assert rig.hangups == ["not-a-fit"]


def test_interrupted_announcement_is_repeated_and_facts_are_kept() -> None:
    async def scenario() -> Rig:
        class Slow(FakeTTS):
            async def stream(self, text: str):  # type: ignore[no-untyped-def]
                self.spoken.append(text)
                for _ in range(self.frames):
                    yield FRAME
                    await asyncio.sleep(0.005)

        llm = FakeLLM([reply("Thanks.", owns_home="no"), reply("We only install on houses you own."), reply("As I said, only owned houses.")])
        rig = Rig(llm, Slow(frames_per_sentence=400))
        rig.pacer.start()
        rig.event("EndOfTurn", "I rent.")
        await until(lambda: len(rig.frames) >= 3)
        rig.event("StartOfTurn", "what")  # caller interrupts the explanation
        await asyncio.sleep(0.02)
        slots_after_interrupt = dict(rig.agent.slots)
        announced_after_interrupt = rig.agent.announced
        rig.tts.frames = 2
        rig.event("EndOfTurn", "What did you say?")
        await settle(rig)
        await rig.agent.close()
        rig.snapshot = (slots_after_interrupt, announced_after_interrupt)  # type: ignore[attr-defined]
        return rig

    rig = run(scenario)
    slots_after_interrupt, announced_after_interrupt = rig.snapshot  # type: ignore[attr-defined]
    assert slots_after_interrupt["owns_home"] == "no"  # what the caller told us stays known
    assert announced_after_interrupt is None  # but they did not hear the whole announcement
    assert "We cannot help this caller" in rig.llm.directive(2)  # so it is announced again
    assert rig.bot_lines()[-1].endswith(ANYTHING_ELSE_QUESTION) and rig.agent.announced == "not_fit"


def test_anything_else_is_spoken_from_the_phrase_cache_not_by_the_model() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM([reply("Thanks.", owns_home="no"), reply("We only install on houses you own.")]))
        rig.pacer.start()
        await rig.say("I rent.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert ANYTHING_ELSE_QUESTION in rig.phrases.served and ANYTHING_ELSE_QUESTION not in rig.tts.spoken
    assert rig.agent.summary()["tts_chars"] == len("We only install on houses you own.")


# ====================================================================== structured output client
def groq(handler: Callable[[httpx.Request], httpx.Response]) -> GroqLLM:
    return GroqLLM(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "groq-test-key", "openai/gpt-oss-120b")


def completion(content: str, **usage: int) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}], "usage": usage or {"prompt_tokens": 400, "completion_tokens": 40}}


def test_reply_json_uses_strict_structured_output() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        body = completion('{"slots": {"owns_home": "no", "monthly_bill": null, "name": null}, "say": "Thanks."}')
        return httpx.Response(200, json=body, headers={"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "7000"})

    async def scenario() -> tuple[dict[str, Any], LLMUsage, GroqLLM]:
        llm = groq(handler)
        usage = LLMUsage()
        out = await llm.reply_json([{"role": "user", "content": "hi"}], reply_schema(DEFS), usage)
        return out, usage, llm

    out, usage, llm = run(scenario)
    assert out == {"slots": {"owns_home": "no", "monthly_bill": None, "name": None}, "say": "Thanks."}
    (body,) = seen
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "reply", "strict": True, "schema": reply_schema(DEFS)},
    }
    assert body["stream"] is False and "stream_options" not in body
    assert body["include_reasoning"] is False and body["reasoning_effort"] == "low"
    assert (usage.requests, usage.prompt_tokens, usage.completion_tokens) == (1, 400, 40)
    assert llm.budget.remaining == 7000


def test_reply_json_falls_back_to_json_mode_and_checks_the_result_itself() -> None:
    formats: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        formats.append(body["response_format"]["type"])
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, json={"error": {"message": "Generated JSON does not match the expected schema."}})
        return httpx.Response(200, json=completion('{"say": "Hello there.", "slots": {}}'))

    async def scenario() -> tuple[dict[str, Any], LLMUsage]:
        usage = LLMUsage()
        return await groq(handler).reply_json([{"role": "user", "content": "hi"}], reply_schema(DEFS), usage), usage

    out, usage = run(scenario)
    assert formats == ["json_schema", "json_object"]
    assert out["say"] == "Hello there." and usage.requests == 2


@pytest.mark.parametrize("bad", ["not json at all", '["a list"]', '{"slots": {}}', '{"say": ""}', '{"say": 42}'])
def test_reply_json_unusable_three_times_is_an_error(bad: str) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=completion(bad))

    async def scenario() -> None:
        with pytest.raises(LLMError, match="no usable reply after retry"):
            await groq(handler).reply_json([{"role": "user", "content": "hi"}], reply_schema(DEFS))

    run(scenario)
    assert calls == 3  # strict attempt, then two retries in plain JSON mode


def test_reply_json_rate_limit_and_network_errors() -> None:
    async def scenario() -> None:
        limited = groq(lambda r: httpx.Response(429, json={"error": {"message": "Rate limit reached"}},
                                                headers={"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "50"}))
        with pytest.raises(LLMError, match="429"):
            await limited.reply_json([], reply_schema(DEFS))
        assert limited.budget.remaining == 0.0

        def offline(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("slow")

        with pytest.raises(LLMError, match="ConnectTimeout"):
            await groq(offline).reply_json([], reply_schema(DEFS))

    run(scenario)


def test_hangup_cancelled_at_the_instant_the_closing_audio_ends_stays_cancelled() -> None:
    """A cancel that lands exactly when the audio queue empties used to be lost
    (asyncio.wait_for swallows it on Python 3.10) and the call was cut anyway."""

    async def scenario() -> Rig:
        llm = FakeLLM([reply("Thanks.", owns_home="no"), reply("We only install on houses you own."), reply("Sure.")])
        rig = Rig(llm, end_silence_s=0.05)
        rig.pacer.start()
        await rig.say("I rent.")
        rig.event("StartOfTurn", "No")
        rig.event("EndOfTurn", "No thanks.")
        await until(lambda: rig.pacer.speaking and len(rig.of("ending")) == 1)  # closing line is playing
        rig.event("StartOfTurn", "Wait")  # barge-in empties the queue and cancels the hang-up in the same instant
        await asyncio.sleep(0.3)  # six silence windows
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.hangups == []
    assert rig.kinds().count("hangup_cancelled") == 1 and rig.agent.summary()["outcome"] is None


def test_reply_json_survives_two_transient_generation_failures() -> None:
    """Seen live: Groq answered 400 'Failed to generate JSON' twice in a row for a
    request that succeeded when sent again."""
    formats: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        formats.append(body["response_format"]["type"])
        assert body["max_completion_tokens"] >= 400  # headroom for slots + sentence + hidden reasoning
        if len(formats) < 3:
            return httpx.Response(400, json={"error": {"message": "Failed to generate JSON.", "failed_generation": "{\"slots\": {"}})
        return httpx.Response(200, json=completion('{"slots": {}, "say": "One to two days."}'))

    async def scenario() -> dict[str, Any]:
        return await groq(handler).reply_json([{"role": "user", "content": "hi"}], reply_schema(DEFS))

    assert run(scenario)["say"] == "One to two days."
    assert formats == ["json_schema", "json_object", "json_object"]
