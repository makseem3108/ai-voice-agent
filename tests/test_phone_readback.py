"""Call 2 on 702 (17 Nov): a complete number was stored without being read back.

The rule had been "read a number back once per call". An incomplete number used
that one time up, so the complete number that followed was kept unheard - and
after a correction as well. Now every complete number is read back and kept
only after the caller's "yes"; the re-ask for an incomplete number is separate."""

from __future__ import annotations

import dataclasses
from typing import Any

from app.agent import MAX_PHONE_READBACKS, fixed_phrases
from app.call_log import format_record
from app.lang import ENGLISH
from app.slots import UNCLEAR, SlotDef
from tests.helpers import NAME, PROFILE, FakeLLM, FakePhrases, Rig, reply, run

PHONE = SlotDef(name="phone", type="phone", question="What is the best phone number?")
BOOKING = dataclasses.replace(PROFILE, slots=(NAME, PHONE), confirmation="Thank you[, {name}]. I've noted your request.")
GOOD, OTHER, SHORT = "0501234567", "0529876543", "050123467"


def heard(number: str) -> str:
    return ENGLISH.phone_confirm.format(digits=ENGLISH.read_digits(number))


async def call(replies: list[Any], says: list[str]) -> Rig:
    rig = Rig(FakeLLM(replies), profile=BOOKING, phrases=FakePhrases(known=set(fixed_phrases(BOOKING))))
    rig.agent.slots.update({"name": "Dana"})
    rig.pacer.start()
    for text in says:
        await rig.say(text)
    await rig.agent.close()
    return rig


def test_after_a_re_ask_the_complete_number_is_still_read_back_and_needs_a_yes() -> None:
    replies = [reply("Okay.", name="Dana", phone=SHORT), reply("Okay.", name="Dana", phone=GOOD), reply("Okay.", name="Dana")]
    unanswered = run(lambda: call(replies, [SHORT, GOOD]))
    assert unanswered.bot_lines() == [ENGLISH.phone_incomplete, heard(GOOD)]
    assert unanswered.agent.slots["phone"] is None and unanswered.agent.summary()["valid_phone"] is False  # hung up before "yes"
    confirmed = run(lambda: call(replies, [SHORT, GOOD, "Yes."]))
    assert confirmed.agent.slots["phone"]["number"] == GOOD and confirmed.agent.announced == "qualified"
    assert [r["complete"] for r in confirmed.of("phone_readback")] == [False, True]


def test_no_to_the_read_back_stores_nothing_and_a_later_yes_cannot_bring_it_back() -> None:
    replies = [reply("Okay.", name="Dana", phone=GOOD), reply("Sorry. What is the number?", name="Dana"), reply("Okay.", name="Dana")]
    rig = run(lambda: call(replies, [GOOD, "No.", "Yes."]))
    assert rig.bot_lines()[0] == heard(GOOD)
    assert rig.agent.slots["phone"] is None and len(rig.of("phone_rejected")) == 1
    assert "said no" in (format_record(rig.of("phone_rejected")[0]) or "")


def test_no_with_another_number_reads_that_one_back_and_keeps_it_only_after_yes() -> None:
    replies = [reply("Okay.", name="Dana", phone=GOOD), reply("Okay.", name="Dana", phone=OTHER), reply("Okay.", name="Dana")]
    waiting = run(lambda: call(replies, [GOOD, f"No, it's {OTHER}."]))
    assert waiting.bot_lines() == [heard(GOOD), heard(OTHER)] and waiting.agent.slots["phone"] is None
    done = run(lambda: call(replies, [GOOD, f"No, it's {OTHER}.", "Yes."]))
    assert done.agent.slots["phone"]["number"] == OTHER


def test_a_question_instead_of_yes_does_not_confirm_the_number() -> None:
    replies = [reply("Okay.", name="Dana", phone=GOOD), reply("We open at nine. What is the best phone number?", name="Dana")]
    rig = run(lambda: call(replies, [GOOD, "What are your opening hours?"]))
    assert rig.agent.slots["phone"] is None


def test_saying_the_same_number_again_confirms_it_without_a_second_read_back() -> None:
    replies = [reply("Okay.", name="Dana", phone=GOOD), reply("Okay.", name="Dana", phone=GOOD)]
    rig = run(lambda: call(replies, [GOOD, f"Yes, {GOOD}."]))
    assert rig.agent.slots["phone"]["number"] == GOOD and len(rig.of("phone_readback")) == 1


def test_correction_from_no_number_at_all_is_read_back_too() -> None:
    """Call 2: the first number was never understood; the corrected one came after a re-ask."""
    replies = [
        reply("Okay.", name="Dana", phone=SHORT),
        reply("Okay.", name="Dana", phone=SHORT),  # still short: left open, outcome without a callback number
        reply("Okay.", name="Dana", phone="29876543"),  # the correction, again misheard: asked once more
        reply("Okay.", name="Dana", phone=OTHER),
        reply("Okay.", name="Dana"),
    ]
    says = [SHORT, SHORT, "The number is wrong, 2 9 8 7 6 5 4 3.", "0 5 2 9 8 7 6 5 4 3"]
    before_yes = run(lambda: call(replies, says))
    assert before_yes.agent.slots["phone"] in (None, UNCLEAR) and before_yes.bot_lines()[-1] == heard(OTHER)
    after_yes = run(lambda: call(replies, [*says, "Yes."]))
    assert after_yes.agent.slots["phone"]["number"] == OTHER and after_yes.agent.summary()["valid_phone"] is True


def test_read_backs_are_limited_so_a_bad_line_cannot_loop_forever() -> None:
    numbers = ["0501234567", "0501234568", "0501234569", "0501234560"]
    replies = [reply("Okay.", name="Dana", phone=n) for n in numbers]
    rig = run(lambda: call(replies, [numbers[0], *[f"No, {n}" for n in numbers[1:]]]))
    assert len(rig.of("phone_readback")) == MAX_PHONE_READBACKS == 3
    assert rig.agent.slots["phone"] == UNCLEAR and len(rig.of("phone_gave_up")) == 1
    assert "not stored" in (format_record(rig.of("phone_gave_up")[0]) or "")
