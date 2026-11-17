"""A caller who changes a detail after the request was confirmed.

Turns after the outcome are answered in plain text and read no slots (see
test_plain_answers.py), so a correction would be answered politely and lost.
Two ways it is caught: a number in what the caller said always gets a normal
JSON turn; for anything else the model answers a plain turn with the marker
[UPDATE], which is never spoken - the turn is done again as JSON. A changed
detail is then confirmed with the same fixed sentence, a new phone number is
read back first."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from app.agent import closing_text, fixed_phrases
from app.call_log import format_record
from app.intent import mentions_digits
from app.lang import ENGLISH
from app.prompt_builder import FORMAT_REMINDER, PLAIN_REMINDER, UPDATE_MARKER, asks_for_update, build_directive, without_marker
from app.slots import ANSWER, OUTCOME_QUALIFIED, Plan, SlotDef
from tests.helpers import NAME, PROFILE, FakeLLM, FakePhrases, Rig, reply, run, until

TIME = SlotDef(name="preferred_time", type="text", question="What day and time suit you?")
PHONE = SlotDef(name="phone", type="phone", question="What is the best phone number?")
CONFIRMATION = "Thank you[, {name}]. I've noted your request[ for {preferred_time}]."
BOOKING = dataclasses.replace(PROFILE, slots=(TIME, NAME, PHONE), confirmation=CONFIRMATION)
ANYTHING_ELSE = ENGLISH.anything_else
OLD, NEW = "0521234567", "0529876543"
FIRST = [
    reply("Thanks.", preferred_time="tomorrow at four", name="Dana"),  # two answers at once: worded again
    reply("What is the best phone number?", preferred_time="tomorrow at four", name="Dana"),
    reply("Okay.", preferred_time="tomorrow at four", name="Dana", phone=OLD),  # read back by the code
    reply("Okay.", preferred_time="tomorrow at four", name="Dana"),  # "yes": confirmed by the code
]


async def booked(*after: Any) -> Rig:
    """A call up to "I've noted your request for tomorrow at four. Anything else?"."""
    rig = Rig(FakeLLM([*FIRST, *after]), profile=BOOKING, phrases=FakePhrases(known=set(fixed_phrases(BOOKING))))
    rig.pacer.start()
    await rig.say("Tomorrow at four, my name is Dana.")
    await rig.say("0 5 2 1 2 3 4 5 6 7")
    await rig.say("Yes.")
    assert rig.agent.announced == "qualified" and rig.agent.slots["phone"]["number"] == OLD
    assert rig.bot_lines()[-1] == f"Thank you, Dana. I've noted your request for tomorrow at four. {ANYTHING_ELSE}"
    return rig


# ---------------------------------------------------------------- trigger a: a number was said
@pytest.mark.parametrize(
    "text, language, expected",
    [
        ("my number is 052 987 6543", "en", True),
        ("five four seven", "en", True),
        ("thanks, bye", "en", False),
        ("What are your opening hours?", "en", False),
        ("בעצם המספר הוא אפס חמש ארבע", "he", True),
        ("אפשר בארבע?", "he", True),  # "at four": the prefix is part of the word
        ("תודה רבה, להתראות", "he", False),
        ("בעצם עדיף לי ביום רביעי בבוקר", "he", False),  # "Wednesday" is not a digit
        ("النمرة صفر خمسة أربعة", "ar", True),
        ("شكراً، مع السلامة", "ar", False),
        ("الأحسن إلي يوم الأربعا الصبح", "ar", False),
        ("", "en", False),
    ],
)
def test_digits_are_noticed_written_or_spoken(text: str, language: str, expected: bool) -> None:
    assert mentions_digits(text, language) is expected


def test_phone_correction_is_read_back_then_confirmed_again() -> None:
    async def scenario() -> Rig:
        rig = await booked(
            reply("Sure.", preferred_time="tomorrow at four", name="Dana", phone=NEW),
            reply("Okay.", preferred_time="tomorrow at four", name="Dana"),
        )
        await rig.say("Sorry, my number is wrong. It's 0 5 2 9 8 7 6 5 4 3.")
        assert rig.bot_lines()[-1] == ENGLISH.phone_confirm.format(digits=ENGLISH.read_digits(NEW))
        assert rig.agent.slots["phone"]["number"] == OLD  # not replaced before the caller says yes
        await rig.say("Yes, that's right.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.llm.plain == [False] * 6  # a number was said: a normal JSON turn, no plain request
    assert rig.llm.directive(4).endswith(FORMAT_REMINDER) and "changing or correcting a detail" in rig.llm.directive(4)
    assert rig.agent.slots["phone"]["number"] == NEW
    assert rig.bot_lines()[-1] == f"Thank you, Dana. I've noted your request for tomorrow at four. {ANYTHING_ELSE}"
    assert rig.agent.announced == "qualified" and rig.agent.summary()["valid_phone"] is True
    assert [(r["slots"], r["pending"]) for r in rig.of("correction")] == [([], ["phone"]), (["phone"], [])]  # heard, then confirmed
    assert rig.agent.summary()["unconfirmed_corrections"] == []
    assert "NOT CONFIRMED YET" in format_record(rig.of("correction")[0])
    assert len(rig.of("phone_readback")) == 2


def test_hang_up_before_confirming_keeps_the_old_number_and_marks_it_unconfirmed() -> None:
    async def scenario() -> Rig:
        rig = await booked(reply("Sure.", preferred_time="tomorrow at four", name="Dana", phone=NEW))
        await rig.say("Sorry, my number is wrong. It's 0 5 2 9 8 7 6 5 4 3.")
        assert rig.bot_lines()[-1] == ENGLISH.phone_confirm.format(digits=ENGLISH.read_digits(NEW))
        await rig.agent.close()  # the caller hangs up without a "yes"
        return rig

    rig = run(scenario)
    summary = rig.agent.summary()
    assert rig.agent.slots["phone"]["number"] == OLD and summary["slots"]["phone"]["number"] == OLD
    assert summary["unconfirmed_corrections"] == ["phone"] and summary["valid_phone"] is True
    assert summary["computed_outcome"] == "qualified"
    assert rig.of("correction")[-1]["pending"] == ["phone"] and rig.of("correction")[-1]["slots"] == []


def test_no_to_the_read_back_keeps_the_old_number_and_does_not_end_the_call() -> None:
    async def scenario() -> Rig:
        rig = await booked(
            reply("Sure.", preferred_time="tomorrow at four", name="Dana", phone=NEW),
            reply("Okay, I kept the number we had.", preferred_time="tomorrow at four", name="Dana", phone=NEW),
        )
        await rig.say("Sorry, my number is wrong. It's 0 5 2 9 8 7 6 5 4 3.")
        await rig.say("No.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.hangups == []  # "no" answered the read-back, not "anything else?"
    assert rig.agent.slots["phone"]["number"] == OLD
    assert rig.agent.summary()["unconfirmed_corrections"] == ["phone"]
    assert rig.of("correction")[-1]["unconfirmed"] == ["phone"]
    assert "UNCONFIRMED" in format_record(rig.of("correction")[-1])
    assert len(rig.of("phone_readback")) == 2  # no third read-back of the same number


def test_time_correction_marker_is_never_spoken_and_the_turn_is_redone_as_json() -> None:
    async def scenario() -> Rig:
        rig = await booked(
            f"{UPDATE_MARKER}",
            reply("Sure, Wednesday morning.", preferred_time="Wednesday morning", name="Dana"),
        )
        await rig.say("Actually, Wednesday morning would be better for me.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.llm.plain == [False, False, False, False, True, False]  # plain first, then the same turn as JSON
    assert rig.llm.directive(4).endswith(PLAIN_REMINDER) and rig.llm.directive(5).endswith(FORMAT_REMINDER)
    assert rig.agent.slots["preferred_time"] == "Wednesday morning"
    # The caller hears the fixed confirmation with the new time - not the marker, not the model's wording.
    assert rig.bot_lines()[-1] == f"Thank you, Dana. I've noted your request for Wednesday morning. {ANYTHING_ELSE}"
    assert not any("UPDATE" in line.upper() for line in rig.bot_lines())
    assert not any("UPDATE" in text.upper() for text in rig.tts.spoken)
    assert rig.of("correction") == [{"type": "correction", "turn": 4, "slots": ["preferred_time"], "marker": True, "pending": [], "unconfirmed": []}]
    assert rig.of("turn")[-1]["llm_path"] == "plain+json"
    assert "[UPDATE]" in format_record(rig.of("correction")[0]) and "preferred_time" in format_record(rig.of("correction")[0])


def test_marker_without_a_real_change_just_answers() -> None:
    async def scenario() -> Rig:
        rig = await booked(UPDATE_MARKER, reply("We open at eight.", preferred_time="Tomorrow at four.", name="Dana"))
        await rig.say("When do you open?")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines()[-1] == "We open at eight."  # reworded slot, same detail: no second confirmation
    assert rig.agent.slots["preferred_time"] == "tomorrow at four"
    assert rig.of("correction") == [{"type": "correction", "turn": 4, "slots": [], "marker": True, "pending": [], "unconfirmed": []}]


def test_thanks_bye_triggers_nothing() -> None:
    async def scenario() -> Rig:
        rig = await booked()
        await rig.say("Thanks, bye.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert len(rig.llm.requests) == 4  # no request at all for the goodbye: no JSON turn, no marker
    assert rig.bot_lines()[-1] == closing_text(BOOKING, "qualified")
    assert rig.of("correction") == [] and rig.agent.slots["phone"]["number"] == OLD


def test_ordinary_question_stays_plain_with_one_request() -> None:
    async def scenario() -> Rig:
        rig = await booked("We are open Monday to Friday.")
        await rig.say("What are your opening hours?")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.llm.plain == [False, False, False, False, True] and rig.of("correction") == []
    assert rig.bot_lines()[-1] == "We are open Monday to Friday."


# ---------------------------------------------------------------- the marker itself
@pytest.mark.parametrize("say", ["[UPDATE]", " [update] ", "[UPDATE] Sure, Wednesday.", "UPDATE", "[ UPDATE ]."])
def test_marker_is_recognised(say: str) -> None:
    assert asks_for_update(say)


@pytest.mark.parametrize("say", ["", "We open at eight.", "I can update the clinic for you.", "عنا دوام من الأحد للخميس."])
def test_ordinary_replies_are_not_the_marker(say: str) -> None:
    assert not asks_for_update(say)


def test_marker_is_stripped_from_anything_that_would_be_spoken() -> None:
    assert without_marker("[UPDATE] Sure, Wednesday.") == "Sure, Wednesday."
    assert without_marker("Okay.") == "Okay."


def test_plain_instruction_names_the_marker_and_json_instruction_does_not() -> None:
    plan = Plan(ANSWER, outcome=OUTCOME_QUALIFIED)
    assert UPDATE_MARKER in build_directive(plan, {})
    forced = build_directive(plan, {}, plain=False)
    assert UPDATE_MARKER not in forced and forced.endswith(FORMAT_REMINDER)
