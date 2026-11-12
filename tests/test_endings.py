"""Endings: a goodbye that also carries an answer, the "qualified, but no call
back" outcome, structured money amounts, and the capped wait before hang-up."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import Any

from app.call_log import format_record
from app.llm_groq import LLMError
from app.profiles import ProfileError, parse_profile
from app.prompt_builder import ANYTHING_ELSE_QUESTION, build_directive, build_system_prompt, incomplete_closing
from app.slots import (
    ANNOUNCE,
    OutcomeVariant,
    SlotDef,
    compute_outcome,
    display_value,
    empty_slots,
    make_plan,
    merge_slots,
    parse_outcomes,
    parse_slots,
    reply_schema,
)
from tests.helpers import BILL, FRAME, NAME, OWNS_HOME, PROFILE, FakeLLM, Rig, reply, run, settle, until

import pytest

CALLBACK = SlotDef(name="callback_ok", type="yes_no", question="Is it okay if an advisor calls you back?")
NO_CALLBACK = OutcomeVariant(
    name="qualified_no_callback",
    when=(("callback_ok", "no"),),
    announce="The caller does not want a call back. Do not promise that anyone will call.",
    closing="Thanks, your details are noted and we won't call. Goodbye!",
)
LEAD = dataclasses.replace(PROFILE, slots=(OWNS_HOME, BILL, NAME, CALLBACK), outcomes=(NO_CALLBACK,))
KNOWN = {"owns_home": "yes", "monthly_bill": "500 shekels", "name": "Mohammed"}


def rig_waiting_for_callback_answer(llm: FakeLLM, **kwargs: Any) -> Rig:
    """A call where everything but callback_ok is known and the bot just asked for it."""
    rig = Rig(llm, profile=LEAD, **kwargs)
    rig.agent.slots.update(KNOWN)
    return rig


# ====================================================================== goodbye that carries an answer
def test_real_call_4_yes_thank_you_goodbye_is_qualified() -> None:
    """The caller's exact words: the "yes" answers the call-back question."""

    async def scenario() -> Rig:
        rig = rig_waiting_for_callback_answer(FakeLLM([reply("Okay.", callback_ok="yes", **KNOWN)]))
        rig.pacer.start()
        await rig.say("Yes. Thank you. Goodbye.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert len(rig.llm.requests) == 1 and "The caller is ending the call" in rig.llm.directive(0)
    assert rig.agent.slots["callback_ok"] == "yes"
    assert rig.bot_lines() == [LEAD.closing_qualified]  # the model's "Okay." is never spoken
    assert rig.tts.spoken == []
    assert rig.hangups == ["goal-complete"]
    summary = rig.agent.summary()
    assert summary["outcome"] == summary["computed_outcome"] == "qualified"
    state = rig.of("state")[-1]
    assert state["slots"]["callback_ok"] == "yes" and state["outcome"] == "qualified"
    # the state is recorded before the closing line is chosen, and the turn is timed
    kinds = [r["type"] for r in rig.records if r["type"] in ("state", "ending", "bot")]
    assert kinds == ["state", "ending", "bot"]
    assert rig.of("turn")[-1]["llm_first_token_ms"] is not None


def test_goodbye_with_a_no_to_the_call_back_is_still_a_valid_lead() -> None:
    async def scenario() -> Rig:
        rig = rig_waiting_for_callback_answer(FakeLLM([reply("Okay.", callback_ok="no")]))
        rig.pacer.start()
        await rig.say("No, please don't call me. Bye.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines() == [NO_CALLBACK.closing]
    assert rig.hangups == ["goal-complete"] and rig.agent.summary()["outcome"] == "qualified_no_callback"


def test_goodbye_that_reveals_a_disqualifier_gets_the_not_fit_closing() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM([reply("Okay.", owns_home="no")]), profile=LEAD)
        rig.pacer.start()
        await rig.say("Oh, I rent. Never mind, goodbye.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines() == [LEAD.closing_not_fit] and rig.hangups == ["not-a-fit"]


def test_goodbye_with_nothing_new_stays_incomplete() -> None:
    async def scenario() -> Rig:
        rig = rig_waiting_for_callback_answer(FakeLLM([reply("Okay.")]))
        rig.pacer.start()
        await rig.say("I have to go. Goodbye.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.agent.slots["callback_ok"] is None and rig.agent.slots["name"] == "Mohammed"
    assert rig.bot_lines() == [incomplete_closing(LEAD)] and rig.agent.summary()["outcome"] == "incomplete"


def test_goodbye_still_closes_when_the_model_fails() -> None:
    async def scenario() -> Rig:
        rig = rig_waiting_for_callback_answer(FakeLLM([LLMError("Groq HTTP 500")]))
        rig.pacer.start()
        await rig.say("Yes. Thank you. Goodbye.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.agent.errors == 1 and rig.of("error")[0]["stage"] == "llm"
    assert rig.bot_lines() == [incomplete_closing(LEAD)]  # closes on what is known; no apology, no retry loop
    assert rig.hangups == ["caller-goodbye"]


def test_caller_keeps_talking_while_the_goodbye_is_being_read() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM([reply("Okay.", callback_ok="yes"), reply("Panels have a 25 year warranty.", callback_ok="yes")])
        llm.hold = asyncio.Event()
        rig = rig_waiting_for_callback_answer(llm)
        rig.pacer.start()
        rig.event("StartOfTurn", "Yes")
        rig.event("EndOfTurn", "Yes. Thank you. Goodbye.")
        await until(lambda: len(llm.requests) == 1)
        rig.event("StartOfTurn", "Oh")  # not done after all
        llm.hold.set()
        await asyncio.sleep(0.05)
        assert rig.hangups == [] and rig.of("ending") == [] and rig.bot_lines() == []
        await rig.say("Oh wait, how long is the warranty?")
        await asyncio.sleep(0.2)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.llm.cancelled == 1 and rig.hangups == []
    # both sentences reach the model together, so the "yes" is not lost
    assert rig.llm.requests[-1][-2]["content"] == "Yes. Thank you. Goodbye. Oh wait, how long is the warranty?"
    assert rig.agent.slots["callback_ok"] == "yes"


def test_decline_after_the_outcome_was_announced_needs_no_model() -> None:
    """Once the caller has been told the outcome, a decline closes at once."""

    async def scenario() -> Rig:
        llm = FakeLLM([reply("Thanks.", callback_ok="yes"), reply("An advisor will call you.")])
        rig = rig_waiting_for_callback_answer(llm)
        rig.pacer.start()
        await rig.say("Yes, that's fine.")
        await rig.say("No. Thank you.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert len(rig.llm.requests) == 2
    assert rig.bot_lines() == [f"An advisor will call you. {ANYTHING_ELSE_QUESTION}", LEAD.closing_qualified]


# ====================================================================== qualified_no_callback
def test_no_call_back_is_announced_asked_once_and_closed_without_a_promise() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM(
            [
                reply("Thanks.", callback_ok="no"),  # written under the old plan
                reply("No problem, your details are noted. Is there anything else?"),
                reply("Panels have a 25 year warranty."),
            ]
        )
        rig = rig_waiting_for_callback_answer(llm)
        rig.pacer.start()
        await rig.say("No, I'd rather you didn't call.")
        await rig.say("What is the warranty?")
        assert rig.hangups == [] and rig.of("ending") == []
        await rig.say("No, that's all.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert NO_CALLBACK.announce in rig.llm.directive(1)
    assert rig.bot_lines() == [
        f"No problem, your details are noted. {ANYTHING_ELSE_QUESTION}",
        "Panels have a 25 year warranty.",
        NO_CALLBACK.closing,
    ]
    assert rig.hangups == ["goal-complete"]
    summary = rig.agent.summary()
    assert summary["outcome"] == "qualified_no_callback" and summary["slots"]["callback_ok"] == "no"


def test_outcome_variants_only_apply_to_complete_fitting_leads() -> None:
    defs, variants = LEAD.slots, LEAD.outcomes
    slots = empty_slots(defs)
    assert compute_outcome(defs, {**slots, "callback_ok": "no"}, variants) is None  # still collecting
    assert compute_outcome(defs, {**slots, **KNOWN, "callback_ok": "no"}, variants) == "qualified_no_callback"
    assert compute_outcome(defs, {**slots, **KNOWN, "callback_ok": "yes"}, variants) == "qualified"
    assert compute_outcome(defs, {**slots, **KNOWN, "owns_home": "no", "callback_ok": "no"}, variants) == "not_fit"
    plan = make_plan(defs, {**slots, **KNOWN, "callback_ok": "no"}, None, variants)
    assert plan.kind == ANNOUNCE and plan.note == NO_CALLBACK.announce
    text = build_directive(plan, {**KNOWN, "callback_ok": "no"})
    assert NO_CALLBACK.announce in text and "what happens next" not in text


def test_parse_outcomes_accepts_a_valid_section() -> None:
    variants, problems = parse_outcomes(
        {"qualified_no_callback": {"when": {"callback_ok": False}, "announce": "Do  not promise a call.", "closing": "Bye!"}},
        LEAD.slots,
    )
    assert problems == []
    assert variants == (OutcomeVariant("qualified_no_callback", (("callback_ok", "no"),), "Do not promise a call.", "Bye!"),)
    assert parse_outcomes(None, LEAD.slots) == ((), [])


@pytest.mark.parametrize(
    "section, message",
    [
        ([], "'outcomes' must be a mapping"),
        ({"qualified": {"when": {"callback_ok": "no"}, "announce": "a", "closing": "b"}}, "this name is built in"),
        ({"Bad Name": {}}, "lowercase letters"),
        ({"x": {"when": {}, "announce": "a", "closing": "b"}}, "'when' must map at least one slot"),
        ({"x": {"when": {"nope": "no"}, "announce": "a", "closing": "b"}}, "unknown slot 'nope'"),
        ({"x": {"when": {"name": "Al"}, "announce": "a", "closing": "b"}}, "needs a 'yes_no' or 'choice' slot"),
        ({"x": {"when": {"callback_ok": "maybe"}, "announce": "a", "closing": "b"}}, "'maybe' is not an allowed value"),
        ({"x": {"when": {"owns_home": "no"}, "announce": "a", "closing": "b"}}, "is disqualifying"),
        ({"x": {"when": {"callback_ok": "no"}, "closing": "b"}}, "'announce' must be non-empty text"),
        ({"x": {"when": {"callback_ok": "no"}, "announce": "a", "closing": ""}}, "'closing' must be non-empty text"),
        ({"x": {"when": {"callback_ok": "no"}, "announce": "a", "closing": "b", "extra": 1}}, "unknown field(s): extra"),
    ],
)
def test_parse_outcomes_reports_problems(section: Any, message: str) -> None:
    variants, problems = parse_outcomes(section, LEAD.slots)
    assert any(message in p for p in problems), problems


def test_profile_with_a_bad_outcome_fails_to_load() -> None:
    data = {
        "business_name": "Acme",
        "agent_role": "the receptionist",
        "goal": "Take a message.",
        "greeting": "Hello.",
        "slots": [{"name": "ok", "type": "yes_no", "question": "Okay?"}],
        "outcomes": {"x": {"when": {"missing": "no"}, "announce": "a", "closing": "b"}},
        "knowledge": "Nothing.",
        "tone": "friendly",
        "closing_qualified": "Bye!",
        "closing_not_fit": "Sorry, bye!",
    }
    with pytest.raises(ProfileError, match="unknown slot 'missing'"):
        parse_profile("t", data)
    data["outcomes"] = {"x": {"when": {"ok": "no"}, "announce": "a", "closing": "See you!"}}
    assert parse_profile("t", data).outcomes[0].closing == "See you!"


# ====================================================================== money slots
MONEY = SlotDef(name="monthly_bill", type="money", question="How much is your bill?", currency="ILS")
MDEFS = (MONEY, NAME)


def test_money_slot_keeps_the_words_and_adds_amount_and_currency() -> None:
    slots, rejected = merge_slots(MDEFS, empty_slots(MDEFS), {"monthly_bill": "over five hundred shekels per month", "name": None})
    assert rejected == []
    assert slots["monthly_bill"] == {"text": "over five hundred shekels per month", "amount": 500, "currency": "ILS"}


def test_money_slot_assumes_the_profile_currency_when_none_was_named() -> None:
    slots, _ = merge_slots(MDEFS, empty_slots(MDEFS), {"monthly_bill": "about 450.50"})
    assert slots["monthly_bill"] == {"text": "about 450.50", "amount": 450.5, "currency": "ILS"}
    slots, _ = merge_slots(MDEFS, empty_slots(MDEFS), {"monthly_bill": "150 dollars"})
    assert slots["monthly_bill"] == {"text": "150 dollars", "amount": 150, "currency": "USD"}


def test_money_slot_without_a_usable_number_keeps_the_raw_text() -> None:
    for words in ("quite a lot", "between 400 and 500 shekels"):  # no number, or two different ones: never guess
        slots, rejected = merge_slots(MDEFS, empty_slots(MDEFS), {"monthly_bill": words})
        assert slots["monthly_bill"] == {"text": words, "amount": None, "currency": None} and rejected == []


def test_money_slot_unknown_stays_empty_and_null_never_erases() -> None:
    empty, rejected = merge_slots(MDEFS, empty_slots(MDEFS), {"monthly_bill": None})
    assert empty["monthly_bill"] is None and rejected == []
    assert merge_slots(MDEFS, empty_slots(MDEFS), {"monthly_bill": "unknown"})[0]["monthly_bill"] is None
    known = {"monthly_bill": {"text": "500 shekels", "amount": 500, "currency": "ILS"}, "name": None}
    assert merge_slots(MDEFS, known, {"monthly_bill": None})[0] == known
    assert merge_slots(MDEFS, known, {"monthly_bill": "500 shekels"})[0] == known  # the model repeats it every turn
    fixed, _ = merge_slots(MDEFS, known, {"monthly_bill": "sorry, 700"})  # a correction replaces it
    assert fixed["monthly_bill"] == {"text": "sorry, 700", "amount": 700, "currency": "ILS"}


def test_money_slot_is_one_plain_text_field_for_the_model() -> None:
    """Regression: three reply fields per money slot made the model stop answering in JSON."""
    schema = reply_schema(MDEFS)["properties"]["slots"]
    assert schema["required"] == ["monthly_bill", "name"]
    assert schema["properties"]["monthly_bill"] == {"type": ["string", "null"]}
    prompt = build_system_prompt(dataclasses.replace(PROFILE, slots=MDEFS))
    assert "- monthly_bill: short text, the amount as the caller said it" in prompt
    assert "monthly_bill_amount" not in prompt and "monthly_bill_currency" not in prompt
    stored = {"monthly_bill": {"text": "five hundred shekels", "amount": 500, "currency": "ILS"}, "name": None}
    directive = build_directive(make_plan(MDEFS, stored, None), stored)
    assert 'Known so far: {"monthly_bill": "five hundred shekels"}' in directive  # the model only sees the words


def test_money_slot_log_line() -> None:
    assert display_value({"text": "five hundred shekels", "amount": 500, "currency": "ILS"}) == '500 ILS ("five hundred shekels")'
    assert display_value({"text": "a lot", "amount": None, "currency": None}) == '"a lot" (amount not understood)'
    line = format_record({"type": "state", "slots": {"monthly_bill": {"text": "x", "amount": 5, "currency": "ILS"}, "name": None}, "phase": "collect"})
    assert line is not None and 'monthly_bill=5 ILS ("x")' in line


def test_money_slot_validation() -> None:
    ok = {"name": "bill", "type": "money", "question": "How much?", "currency": "ILS"}
    assert parse_slots([ok])[1] == []
    assert any("needs 'currency'" in p for p in parse_slots([{**ok, "currency": "shekels"}])[1])
    assert any("needs 'currency'" in p for p in parse_slots([{k: v for k, v in ok.items() if k != "currency"}])[1])
    assert any("only allowed for type 'money'" in p for p in parse_slots([{**ok, "type": "text"}])[1])


# ====================================================================== hang-up wait
class Noise:
    """A voice detector that hears something in every frame."""

    last_rms = 2000.0

    def feed(self, frame: bytes) -> bool:
        return True

    def reset(self) -> None:
        pass


def _close_and_time(noisy: bool) -> tuple[Rig, float]:
    async def scenario() -> tuple[Rig, float]:
        llm = FakeLLM([reply("Thanks.", callback_ok="yes"), reply("An advisor will call you.")])
        rig = rig_waiting_for_callback_answer(llm, detector=Noise(), end_silence_s=0.1, end_silence_max_s=0.3)
        rig.pacer.start()
        await rig.say("Yes, that's fine.")
        await rig.say("No. Thank you.")
        await asyncio.wait_for(rig.pacer.drained(), 2)
        closing_done = time.monotonic()
        while not rig.hangups:
            if noisy:
                rig.agent.on_caller_audio(FRAME)
            assert time.monotonic() - closing_done < 2.0, "the line was held open past the limit"
            await asyncio.sleep(0.005)
        waited = time.monotonic() - closing_done
        await rig.agent.close()
        return rig, waited

    return run(scenario)


def test_hang_up_comes_after_the_short_silence_on_a_quiet_line() -> None:
    rig, waited = _close_and_time(noisy=False)
    assert 0.08 <= waited < 0.25
    assert rig.of("hangup_extended") == []


def test_noise_extends_the_wait_only_up_to_the_cap_and_is_logged() -> None:
    rig, waited = _close_and_time(noisy=True)
    assert 0.25 <= waited < 0.45  # capped at 0.3 s although the noise never stops
    extended = rig.of("hangup_extended")
    assert extended and extended[-1]["capped"] is True
    assert all(r["rms"] == 2000 and r["hangup_in_ms"] <= 100 for r in extended)
    assert len([r for r in extended if r["capped"]]) == 1  # nothing more to log once the limit is reached
    line = format_record(extended[-1])
    assert line is not None and line.startswith("[hangup_extended]") and "(at the limit)" in line


def test_default_wait_is_one_and_a_half_seconds_capped_at_three() -> None:
    from app.agent import Agent
    import inspect

    params = inspect.signature(Agent.__init__).parameters
    assert params["end_silence_s"].default == 1.5 and params["end_silence_max_s"].default == 3.0
