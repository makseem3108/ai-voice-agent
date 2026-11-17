"""Guards added after the first Hebrew / Arabic calls (2026-11-12):
loop guard, "didn't catch that", phone read-back, fixed confirmation sentence, scope wording."""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from app.agent import closing_text, fixed_phrases, hangup_reason
from app.call_log import format_record
from app.intent import is_question
from app.lang import ARABIC, HEBREW
from app.phones import extract_digits, is_valid_israeli, phone_value, spoken_digits
from app.profiles import ProfileError, load_profile, parse_profile
from app.prompt_builder import build_directive, build_system_prompt
from app.rtp_stats import parse_rtp_stats
from app.slots import UNCLEAR, OutcomeVariant, Plan, SlotDef, compute_outcome, display_value, make_plan, merge_slots, missing_slots, render_template
from app.stt_deepgram import Transcript, parse_message
from app.stt_nova import NovaTurnTracker
from tests.helpers import BILL, NAME, OWNS_HOME, PROFILE, FakeLLM, FakePhrases, Rig, reply, run, until

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
STATUS = SlotDef(name="patient_status", type="choice", values=("new", "existing"), question="هاي أول زيارة إلك عنا، ولا إنت متعالج عنا؟", description="new for a first visit")
PHONE = SlotDef(name="phone", type="phone", question="What is the best phone number?")


def clinic(language: str = "ar", **extra: Any) -> Any:
    """A small clinic-like profile: an optional scope slot, a choice, a name, a phone."""
    scope = SlotDef(name="service_offered", type="yes_no", required=False, disqualifying=("no",), not_fit_reason="this is a dental clinic")
    return dataclasses.replace(PROFILE, language=language, slots=(scope, STATUS, NAME, PHONE), **extra)


def rig_for(profile: Any, llm: FakeLLM, **kwargs: Any) -> Rig:
    return Rig(llm, profile=profile, phrases=FakePhrases(known=set(fixed_phrases(profile))), loop_guard=True, **kwargs)


# ====================================================================== loop guard
def test_real_call_702_same_question_is_not_asked_a_third_time() -> None:
    """The Arabic call: five turns, the same question five times. Now: ask, rephrase, move on."""
    question = STATUS.question

    async def scenario() -> Rig:
        llm = FakeLLM(
            [
                reply(question, service_offered="yes"),  # turn 1: first ask
                reply(question),  # turn 2: answer not understood -> discarded, rephrase requested
                reply("ما فهمت عليك. هاي أول مرة بتيجي عنا، ولا إجيت من قبل؟"),
                reply(question),  # turn 3: still nothing -> slot marked unclear, next question requested
                reply("تمام. شو الاسم الكامل لو سمحت؟"),
            ]
        )
        rig = rig_for(clinic(), llm)
        rig.pacer.start()
        await rig.say("مرحبا بدي احجز موعد")
        await rig.say("اونجا راند")
        await rig.say("هيك يعني")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines() == [STATUS.question, "ما فهمت عليك. هاي أول مرة بتيجي عنا، ولا إجيت من قبل؟", "تمام. شو الاسم الكامل لو سمحت؟"]
    rephrase = rig.llm.directive(2)
    assert "did not make it clear" in rephrase and "name the possible answers (new / existing: new for a first visit)" in rephrase
    assert rig.agent.slots["patient_status"] == UNCLEAR
    assert [(g["action"], g["slot"]) for g in rig.of("loop_guard")] == [("rephrase", "patient_status"), ("skip", "patient_status")]
    assert rig.of("state")[-1]["next_slot"] == "name" and rig.of("state")[-1]["unclear"] == ["patient_status"]
    assert '"patient_status": "unclear"' in rig.llm.directive(4)  # the model is told it was given up on
    assert rig.agent.summary()["unclear_slots"] == ["patient_status"]
    assert (format_record(rig.of("loop_guard")[1]) or "").startswith("[loop_guard] patient_status: still unanswered the third time: marked unclear")


def test_an_answer_on_the_second_try_ends_the_guard() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM([reply(STATUS.question, service_offered="yes"), reply(STATUS.question), reply("Sorry? First visit, or been here before?"), reply("Thanks. Your name?", patient_status="new")])
        rig = rig_for(clinic("en"), llm)
        rig.pacer.start()
        await rig.say("Hello, an appointment please.")
        await rig.say("Mmm.")
        await rig.say("First time.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.agent.slots["patient_status"] == "new" and [g["action"] for g in rig.of("loop_guard")] == ["rephrase"]
    assert rig.of("state")[-1]["asked"] == {"name": 1}  # the next slot starts counting from its own first ask


def test_a_caller_who_asks_questions_is_not_treated_as_failing_to_answer() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM([reply(STATUS.question), reply("We are open Sunday to Thursday. " + STATUS.question), reply("A check-up takes 45 minutes. " + STATUS.question), reply("Until five. " + STATUS.question)])
        rig = rig_for(clinic("en"), llm)
        rig.pacer.start()
        await rig.say("Hello, an appointment please.")
        for question in ("When are you open?", "How long does a check-up take?", "Until what time?"):
            await rig.say(question)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.of("loop_guard") == [] and len(rig.llm.requests) == 4 and rig.agent.slots["patient_status"] is None
    assert is_question("When are you open?") and is_question("شو ساعات الدوام", "ar") and is_question("מה השעות", "he") and not is_question("First time.")


def test_a_slot_that_decides_the_outcome_ends_in_a_call_back_instead_of_being_skipped() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM([reply(OWNS_HOME.question), reply(OWNS_HOME.question), reply("Sorry, do you own it, yes or no?"), reply(OWNS_HOME.question)])
        rig = Rig(llm, loop_guard=True, end_silence_s=0.05)  # PROFILE: owns_home can disqualify
        rig.pacer.start()
        await rig.say("Hello.")
        await rig.say("Well.")
        await rig.say("Hmm.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines()[-1] == closing_text(PROFILE, "callback") == "I'm sorry, I'm having trouble understanding. Someone from Acme Solar will call you back. Goodbye!"
    assert rig.hangups == ["needs-callback"] == [hangup_reason("callback")] and rig.agent.summary()["outcome"] == "callback"
    assert rig.agent.slots["owns_home"] is None and rig.of("loop_guard")[-1]["action"] == "callback"
    assert len(rig.llm.requests) == 4  # the third round's reply was never spoken


def test_unclear_counts_as_asked_but_never_disqualifies() -> None:
    defs = clinic().slots
    slots = {"service_offered": None, "patient_status": UNCLEAR, "name": "Dana", "phone": {"text": "0521234567", "number": "0521234567", "valid": True}}
    assert missing_slots(defs, slots) == [] and compute_outcome(defs, slots) == "qualified"
    assert merge_slots(defs, slots, {"patient_status": None})[0]["patient_status"] == UNCLEAR  # null never erases it
    assert merge_slots(defs, slots, {"patient_status": "new"})[0]["patient_status"] == "new"  # a real answer later replaces it


# ====================================================================== "didn't catch that"
def test_text_that_cannot_be_the_callers_language_is_not_sent_to_the_model() -> None:
    """702: Nova-3 returned "Hi. I wanna", "I'm", "Alice Boy." on the Arabic line."""

    async def scenario() -> Rig:
        llm = FakeLLM([reply(STATUS.question, service_offered="yes"), reply("تمام. شو الاسم الكامل لو سمحت؟")])
        rig = rig_for(clinic(), llm)
        rig.pacer.start()
        await rig.say("مرحبا بدي احجز موعد")
        await rig.say("Hi. I wanna")
        assert len(llm.requests) == 1 and rig.bot_lines()[-1] == ARABIC.didnt_catch  # second ask, no model
        await rig.say("Alice Boy.")  # third time: give up on the slot, the model asks the next one
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines() == [STATUS.question, ARABIC.didnt_catch, "تمام. شو الاسم الكامل لو سمحت؟"]
    assert rig.agent.slots["patient_status"] == UNCLEAR and rig.agent.slots["name"] is None  # "Alice Boy" was not stored as a name
    assert [u["text"] for u in rig.of("unintelligible")] == ["Hi. I wanna", "Alice Boy."] and rig.agent.summary()["unintelligible"] == 2
    assert (format_record(rig.of("unintelligible")[0]) or "").startswith("[unintelligible] speech recognition returned text that is not in this line's language")
    assert rig.of("bot")[1]["kind"] == "didnt_catch"


def test_digits_and_the_right_alphabet_pass_and_english_lines_are_never_checked() -> None:
    arabic = rig_for(clinic(), FakeLLM())
    assert arabic.agent._is_unintelligible("Alice Boy.") and arabic.agent._is_unintelligible("Hi")
    assert not arabic.agent._is_unintelligible("0501234567") and not arabic.agent._is_unintelligible("أول مرة")
    assert not arabic.agent._is_unintelligible("بدي תור")  # mixed: some of it is Arabic
    mixed = rig_for(dataclasses.replace(clinic(), stt=dataclasses.replace(PROFILE.stt, language="multi", language_hints=("ar", "he"))), FakeLLM())
    assert not mixed.agent._is_unintelligible("בסדר") and mixed.agent._is_unintelligible("okay")
    english = rig_for(clinic("en"), FakeLLM())
    assert not english.agent._is_unintelligible("שלום") and not english.agent._is_unintelligible("Hello")


def test_hebrew_goodbye_is_still_understood_before_the_check() -> None:
    async def scenario() -> Rig:
        rig = rig_for(clinic("he"), FakeLLM([reply("Okay.")]), end_silence_s=0.05)
        rig.pacer.start()
        await rig.say("להתראות")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    assert run(scenario).of("unintelligible") == []


# ====================================================================== phone numbers
@pytest.mark.parametrize(
    "said, number, valid",
    [
        ("0501234567", "0501234567", True),  # mobile: 05X + 7 digits
        ("050123467", "050123467", False),  # the 701 call: one digit lost
        ("05-012-34567", "0501234567", True),
        ("03 123 4567", "031234567", True),  # landline: 0X + 7 digits
        ("072-1234567", "0721234567", True),
        ("+972 50 123 4567", "0501234567", True),
        ("אפס חמש אפס אחת שתיים שלוש ארבע חמש שש שבע", "0501234567", True),
        ("صفر خمسة صفر واحد اثنين ثلاثة أربعة خمسة ستة سبعة", "0501234567", True),
        ("zero five two, one two three, four five six seven", "0521234567", True),
        ("٠٥٠١٢٣٤٥٦٧", "0501234567", True),
        ("0123", "0123", False),
        ("05012345671", "05012345671", False),
        ("061234567", "061234567", False),
    ],
)
def test_israeli_phone_numbers(said: str, number: str, valid: bool) -> None:
    assert extract_digits(said) == number and is_valid_israeli(number) is valid
    assert phone_value(said) == {"text": said, "number": number, "valid": valid}


def test_phone_slot_values_and_display() -> None:
    assert phone_value(None) is None and phone_value("unknown") is None and phone_value("my number")["number"] is None
    slots, rejected = merge_slots((PHONE,), {"phone": None}, {"phone": "0501234567"})
    assert slots["phone"] == {"text": "0501234567", "number": "0501234567", "valid": True} and rejected == []
    assert display_value(slots["phone"]) == "0501234567" and display_value(phone_value("050123467")) == "050123467 (not a complete number)"
    assert spoken_digits("0501") == "0 5 0 1"
    assert "- phone: the phone number the caller said, as digits" in build_system_prompt(clinic("en"))


def test_real_call_701_incomplete_number_is_read_back_and_asked_once_more() -> None:
    async def scenario(second: str) -> Rig:
        profile = clinic("en")
        llm = FakeLLM(
            [
                reply("Thanks.", phone="050123467"),  # 9 digits: not accepted yet
                reply("Thanks.", phone=second),
                reply("Okay."),
                reply("Thank you, we will call you."),
            ]
        )
        rig = rig_for(profile, llm)
        rig.agent.slots.update({"patient_status": "new", "name": "Dana"})
        rig.pacer.start()
        await rig.say("050123467")
        assert rig.agent.slots["phone"] is None and rig.of("phone_readback")[0]["heard"] == "0 5 0 1 2 3 4 6 7"
        await rig.say(second)
        if rig.bot_lines()[-1].startswith("I heard"):
            assert rig.agent.slots["phone"] is None  # the re-ask did not use up the read-back (call of 17 Nov, 702)
            await rig.say("Yes.")
        await rig.agent.close()
        return rig

    fixed = run(lambda: scenario("0501234567"))
    assert fixed.bot_lines()[0] == "I didn't catch the whole number. Could you say it again, slowly?"  # no digits read back
    assert fixed.bot_lines()[1] == "I heard zero five zero, one two three, four five six seven. Is that right?"
    assert fixed.agent.slots["phone"] == {"text": "0501234567", "number": "0501234567", "valid": True}
    assert (format_record(fixed.of("phone_readback")[0]) or "").startswith('[phone_readback] phone: heard "0 5 0 1 2 3 4 6 7", which is not a whole number; asked for it again')
    assert fixed.agent.summary()["valid_phone"] is True
    still_short = run(lambda: scenario("050123467"))  # asked again once only; a fragment is never kept as the number
    assert still_short.agent.slots["phone"] == UNCLEAR and len(still_short.of("phone_readback")) == 1
    assert still_short.of("phone_invalid")[0]["heard"] == "0 5 0 1 2 3 4 6 7" and still_short.agent.summary()["valid_phone"] is False
    assert (format_record(still_short.of("phone_invalid")[0]) or "").startswith('[phone_invalid] "0 5 0 1 2 3 4 6 7" is still not a phone number')


def test_real_call_step5_a_complete_number_is_read_back_too_and_yes_confirms_it() -> None:
    """The bot stored 0501234576 where the caller said 0501234567: a wrong number can look valid."""

    async def scenario(answer: str, replies: list[Any], then: str = "") -> Rig:
        llm = FakeLLM([reply("Thanks.", phone="0501234576"), *replies])
        rig = rig_for(clinic("en"), llm)
        rig.agent.slots.update({"patient_status": "new", "name": "Dana"})
        rig.pacer.start()
        await rig.say("0501234576")
        assert rig.agent.slots["phone"] is None  # not accepted before the caller has heard it back
        await rig.say(answer)
        if then:
            assert rig.agent.slots["phone"] is None  # the corrected number is read back as well
            await rig.say(then)
        await rig.agent.close()
        return rig

    confirmed = run(lambda: scenario("Yes, that's right.", [reply("Okay."), reply("Thank you, we will call you.")]))
    assert confirmed.bot_lines()[0] == "I heard zero five zero, one two three, four five seven six. Is that right?"
    assert confirmed.agent.slots["phone"]["number"] == "0501234576"  # "yes" alone confirms what was read back
    corrected = run(lambda: scenario("No, it's 0501234567.", [reply("Thanks.", phone="0501234567"), reply("Okay."), reply("Thank you.")], then="Yes."))
    assert corrected.bot_lines()[1] == "I heard zero five zero, one two three, four five six seven. Is that right?"
    assert corrected.agent.slots["phone"]["number"] == "0501234567" and len(corrected.of("phone_readback")) == 2
    refused = run(lambda: scenario("No.", [reply("Sorry. What is the number?")]))
    assert refused.agent.slots["phone"] is None  # a bare "no": ask again, do not keep the wrong number


# ====================================================================== fixed confirmation sentence
def test_templates_fill_slots_and_drop_optional_parts() -> None:
    template = "Thank you[, {name}]. I've noted your request[ for {preferred_time}]."
    assert render_template(template, {"name": "Dana", "preferred_time": "Tuesday at four"}) == "Thank you, Dana. I've noted your request for Tuesday at four."
    assert render_template(template, {"name": None, "preferred_time": UNCLEAR}) == "Thank you. I've noted your request."
    assert render_template("Amount: {bill}.", {"bill": {"text": "500 shekels", "amount": 500, "currency": "ILS"}}) == "Amount: 500 shekels."
    assert render_template("No slots here.", {}) == "No slots here."


def test_real_call_701_confirmation_is_the_profiles_sentence_not_the_models() -> None:
    """The model had said: you are booked for tomorrow at four, arrive ten minutes early for a form."""
    profile = clinic("en", confirmation="Thank you[, {name}]. I've noted your request. The front desk will call you to confirm.")

    async def scenario() -> Rig:
        llm = FakeLLM(
            [
                reply("Thanks.", phone="0501234567"),
                reply("You are booked for tomorrow at four. Please arrive ten minutes early.", phone="0501234567"),
            ]
        )
        rig = rig_for(profile, llm, end_silence_s=0.05)
        rig.agent.slots.update({"patient_status": "existing", "name": "Dana"})
        rig.pacer.start()
        await rig.say("0501234567")
        await rig.say("Yes.")
        await rig.say("No, thank you.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    spoken = rig.bot_lines()[1]
    assert spoken == "Thank you, Dana. I've noted your request. The front desk will call you to confirm. Is there anything else I can help you with?"
    assert "booked" not in " ".join(rig.bot_lines()) and "arrive" not in " ".join(rig.bot_lines())
    assert len(rig.llm.requests) == 2  # one per turn: neither the read-back nor the announcement is worded by the model
    assert rig.of("state")[-2]["templated"] is True and rig.of("state")[-2]["outcome"] == "qualified"
    assert rig.hangups == ["goal-complete"]


def test_profiles_without_a_template_keep_the_models_announcement() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM([reply("Thanks.", name="Dana"), reply("Thank you, we will be in touch.")])
        rig = rig_for(dataclasses.replace(clinic("en"), slots=(STATUS, NAME)), llm)
        rig.agent.slots.update({"patient_status": "existing"})
        rig.pacer.start()
        await rig.say("Dana.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines()[0].startswith("Thank you, we will be in touch.") and len(rig.llm.requests) == 2


def test_template_must_only_use_slots_that_exist() -> None:
    base = {"business_name": "Acme", "agent_role": "r", "goal": "g", "greeting": "Hello.", "knowledge": "k", "tone": "t",
            "slots": [{"name": "ok", "type": "yes_no", "question": "Okay?"}], "closing_qualified": "Bye!", "closing_not_fit": "Sorry, bye!"}
    assert parse_profile("t", {**base, "confirmation": "Noted[: {ok}]."}).confirmation == "Noted[: {ok}]."
    with pytest.raises(ProfileError, match="'confirmation': unknown slot\\(s\\) in the text: when"):
        parse_profile("t", {**base, "confirmation": "Booked for {when}."})
    variant = {"x": {"when": {"ok": "no"}, "announce": "a", "closing": "b", "confirmation": "Noted, no call."}}
    assert parse_profile("t", {**base, "outcomes": variant}).outcomes[0] == OutcomeVariant("x", (("ok", "no"),), "a", "b", "Noted, no call.")


@pytest.mark.parametrize("name", ["clinic", "clinic_he", "clinic_ar", "lead_qualification"])
def test_shipped_profiles_confirm_with_a_fixed_sentence_that_promises_no_booking(name: str) -> None:
    profile = load_profile(name, PROFILES_DIR)
    assert profile.confirmation
    full = render_template(profile.confirmation, {slot.name: "X" for slot in profile.slots})
    bare = render_template(profile.confirmation, {})
    for text in (full, bare):
        assert text and "[" not in text and "{" not in text
        for word in ("booked", "confirmed for", "נקבע", "מאושר", "تم الحجز", "مؤكد"):
            assert word not in text
    if name == "clinic_ar":
        assert "ق" not in profile.confirmation


def test_system_prompt_forbids_commitments_and_stays_static() -> None:
    prompt = build_system_prompt(PROFILE)
    assert "Never say that anything is booked, confirmed, scheduled or agreed" in prompt and "never promise a date, a time or a price" in prompt
    assert prompt == build_system_prompt(PROFILE)


# ====================================================================== scope wording
@pytest.mark.parametrize("name", ["clinic", "clinic_he", "clinic_ar"])
def test_clinic_profiles_spell_out_what_is_out_of_scope(name: str) -> None:
    profile = load_profile(name, PROFILES_DIR)
    scope = next(slot for slot in profile.slots if slot.name == "service_offered")
    assert "This is a dental clinic" in scope.description and "a family doctor, an eye exam" in scope.description
    assert scope.disqualifying == ("no",) and "this is a dental clinic" in scope.not_fit_reason
    assert next(slot for slot in profile.slots if slot.name == "phone").type == "phone"
    slots = {slot.name: None for slot in profile.slots}
    plan = make_plan(profile.slots, {**slots, "service_offered": "no"}, None, profile.outcomes)
    directive = build_directive(plan, {"service_offered": "no"})
    assert plan.outcome == "not_fit" and "this is a dental clinic" in directive and "says what kind of business this is" in directive


# ====================================================================== diagnostics
def test_detected_language_is_logged_in_multi_mode() -> None:
    message = '{"type":"Results","is_final":true,"speech_final":true,"start":0,"duration":1,"channel":{"alternatives":[{"transcript":"Hi. I wanna","confidence":0.6,"languages":["en"],"words":[]}]}}'
    parsed = parse_message(message)
    assert isinstance(parsed, Transcript) and parsed.languages == ("en",)
    tracker = NovaTurnTracker()
    end = tracker.feed(parsed)[-1]
    assert end.details["languages"] == ["en"]
    line = format_record({"type": "stt_event", "event": "EndOfTurn", "details": end.details})
    assert line is not None and "detected=en" in line
    mono = NovaTurnTracker().feed(Transcript("أول مرة", True, True, 0.0, 1.0))[-1]
    assert "languages" not in mono.details  # single-language mode sends none


def test_negative_rtcp_loss_counter_is_not_shown_as_millions() -> None:
    """702: Asterisk printed rlp=16777215 (a signed 24-bit minus one) and we showed 99.99 % loss."""
    stats = parse_rtp_stats("lp=0;rxcount=2883;txcount=1587;rlp=16777215")
    assert stats["tx_lost"] == 0 and stats["tx_loss_pct"] == 0.0
    assert parse_rtp_stats("txcount=1000;rlp=16777214")["tx_lost"] == 0 and parse_rtp_stats("txcount=1000;rlp=12")["tx_lost"] == 12


def test_new_fixed_phrases_exist_in_every_language() -> None:
    for pack in (HEBREW, ARABIC):
        assert pack.didnt_catch and "{business}" in pack.callback_closing and "ق" not in ARABIC.didnt_catch + ARABIC.callback_closing
    assert closing_text(load_profile("clinic_he", PROFILES_DIR), "callback") == "מצטערת, הקו לא כל כך ברור. מישהו ממרפאת השיניים הרבור יחזור אליכם בטלפון. להתראות!"
    assert closing_text(load_profile("clinic_ar", PROFILES_DIR), "callback") == "آسفة، الصوت مش واضح عندي. رح نتصل فيك من عيادة الشفاء لطب الأسنان. مع السلامة!"


# ====================================================================== follow-ups from the Step 5 calls
def test_real_call_step5_one_misheard_word_does_not_end_the_booking() -> None:
    """ "بدي أحجز תור" was heard as "بدي احجز صوره" and the model set service_offered=no: not-a-fit at once."""

    async def scenario(answer: str, replies: list[Any]) -> Rig:
        llm = FakeLLM([reply("Okay.", service_offered="no"), reply("بدك تحجز صورة؟ فهمت عليك صح؟"), *replies])
        rig = rig_for(clinic(), llm, end_silence_s=0.05)
        rig.pacer.start()
        await rig.say("بدي احجز صوره يا دكتور")
        assert rig.agent.slots["service_offered"] is None and rig.of("state")[-1]["outcome"] is None  # not decided yet
        await rig.say(answer)
        await rig.agent.close()
        return rig

    corrected = run(lambda: scenario("لا، بدي موعد عند الدكتور", [reply(STATUS.question, service_offered="yes")]))
    check = corrected.llm.directive(1)
    assert "Do not refuse yet" in check and "says who you are and what this business does" in check and "this is a dental clinic" in check
    assert corrected.bot_lines() == ["بدك تحجز صورة؟ فهمت عليك صح؟", STATUS.question]
    assert corrected.agent.slots["service_offered"] == "yes" and len(corrected.of("scope_check")) == 1
    confirmed = run(lambda: scenario("آه، صورة أشعة للصدر", [reply("Okay.", service_offered="no"), reply("إحنا عيادة أسنان وما بنعمل صور أشعة للصدر.")]))
    assert confirmed.agent.slots["service_offered"] == "no" and confirmed.of("state")[-1]["outcome"] == "not_fit"
    assert len(confirmed.of("scope_check")) == 1  # asked once, then believed
    assert (format_record(confirmed.of("scope_check")[0]) or "").startswith("[scope_check] service_offered: it sounded like something we do not offer")


def test_a_disqualifying_answer_to_a_question_we_asked_needs_no_second_check() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM([reply("Thanks.", owns_home="no"), reply("We only install on houses you own.")]), loop_guard=True)
        rig.pacer.start()
        await rig.say("I rent an apartment.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.of("scope_check") == [] and rig.of("state")[-1]["outcome"] == "not_fit"


def test_real_call_step5_answering_a_different_question_is_progress_not_a_failed_answer() -> None:
    """The bot asked for the reason; the caller said "أول مرة" (patient status). That was treated as unanswered."""
    reason = SlotDef(name="reason", type="text", question="شو سبب الزيارة؟")

    async def scenario() -> Rig:
        llm = FakeLLM([reply(reason.question), reply(reason.question, patient_status="new"), reply("تمام. " + STATUS.question[:0] + "شو الاسم الكامل؟", reason="تنظيف اسنان")])
        rig = rig_for(dataclasses.replace(clinic(), slots=(reason, STATUS, NAME)), llm)
        rig.pacer.start()
        await rig.say("مرحبا")
        await rig.say("أول مرة")
        assert rig.of("loop_guard") == [] and rig.agent.slots["patient_status"] == "new"
        assert rig.of("state")[-1]["asked"] == {"reason": 1}  # still its first ask as far as the guard goes
        await rig.say("تنظيف أسنان")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.of("loop_guard") == [] and len(rig.llm.requests) == 3 and rig.of("state")[-1]["asked"] == {"name": 1}


def test_listening_notes_reach_the_model_only_for_profiles_that_have_them() -> None:
    arabic = load_profile("clinic_ar", PROFILES_DIR)
    prompt = build_system_prompt(arabic)
    assert "Callers mix Hebrew words into their Arabic" in prompt and "most likely means תור (an appointment)" in prompt
    assert {"كوبات حوليم", "كلاليت", "هفنياه", "الدوام"} <= set(arabic.keyterms) and "تور" not in arabic.keyterms
    assert arabic.stt.language == "" and arabic.stt.language_hints == ()  # Arabic only; multi mode is off
    assert "Callers mix" not in build_system_prompt(load_profile("clinic_he", PROFILES_DIR))
    with pytest.raises(ProfileError, match="listening_notes"):
        parse_profile("t", {"listening_notes": 3})
