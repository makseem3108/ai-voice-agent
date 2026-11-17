"""After the calls of 2026-11-12 (evening): a caller who is not calling about what the
business does, a caller who would rather not give a number, and the phone read-back.

On the Arabic line a caller asked for tomatoes. The text was readable, but the model
never marked it "not for us", so the bot kept asking its clinic questions and the loop
guard moved on to the next one. Now: one polite "this is what we are - is that what
you need?", then a polite not-a-fit closing. Never the next slot question."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.agent import closing_text, fixed_phrases
from app.call_log import format_record
from app.intent import agrees, skips_question
from app.lang import ARABIC, ENGLISH, HEBREW
from app.llm_groq import TokenLedger, tokens_used
from app.profiles import load_profile
from app.prompt_builder import build_system_prompt
from app.slots import UNCLEAR, SlotDef
from tests.helpers import NAME, PROFILE, FakeLLM, FakePhrases, Rig, reply, run, until
from tests.test_llm_fallback_model import BIG, SMALL, TPD, TPM, ask, limited, llm, ok

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
SCOPE = SlotDef(name="service_offered", type="yes_no", required=False, disqualifying=("no",), not_fit_reason="this is a dental clinic")
REASON = SlotDef(name="reason", type="text", question="شو سبب الزيارة؟ مثلاً فحص، تنظيف، أو وجع أسنان.")
STATUS = SlotDef(name="patient_status", type="choice", values=("new", "existing"), question="Is this your first visit, or have you been here before?")
PHONE = SlotDef(name="phone", type="phone", question="What is the best phone number?")
REDIRECT = "أنا مساعدة عيادة الأسنان. بدك تحجز موعد عنا؟"


def clinic(language: str = "ar", slots: tuple[SlotDef, ...] = (SCOPE, REASON, STATUS, NAME, PHONE)) -> Any:
    return dataclasses.replace(PROFILE, language=language, slots=slots)


def rig_for(profile: Any, llm_: FakeLLM, **kwargs: Any) -> Rig:
    return Rig(llm_, profile=profile, phrases=FakePhrases(known=set(fixed_phrases(profile))), loop_guard=True, end_silence_s=0.05, **kwargs)


# ====================================================================== off-topic callers
def test_real_call_702_tomatoes_get_one_redirect_then_a_polite_not_a_fit_closing() -> None:
    async def scenario() -> Rig:
        llm_ = FakeLLM(
            [
                reply(REASON.question),  # turn 1: garbled, nothing to go on yet
                reply("معك عيادة الشفاء، ممكن توضح شو بدك بالضبط؟"),  # turn 2: still nothing -> say what we are instead
                reply(REDIRECT),
                reply(STATUS.question),  # turn 3: the model would ask the next thing; never spoken
            ]
        )
        rig = rig_for(clinic(), llm_)
        rig.pacer.start()
        await rig.say("بد 4 كيلوبات")
        await rig.say("بدي بندوره بقول")
        await rig.say("فش الزياره بدعش اضرب ولا دكتور")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines() == [REASON.question, REDIRECT, closing_text(clinic(), "not_fit")]
    assert STATUS.question not in rig.bot_lines()  # never the next slot question
    assert [(g["action"], g["slot"]) for g in rig.of("loop_guard")] == [("redirect", "reason"), ("off_topic", "reason")]
    directive = rig.llm.directive(2)
    assert "says who you are and what this business does" in directive and "Do not refuse yet" in directive
    assert rig.hangups == ["not-a-fit"] and rig.agent.summary()["outcome"] == "not_fit"
    assert all(value is None for value in rig.agent.slots.values())  # nothing was made up about the caller
    assert (format_record(rig.of("loop_guard")[0]) or "").startswith("[loop_guard] reason: nothing learned yet: said what this business is")
    assert (format_record(rig.of("loop_guard")[1]) or "").startswith("[loop_guard] reason: still nothing about what we do after that")


def test_when_the_model_spots_it_the_redirect_comes_on_the_first_turn() -> None:
    async def scenario() -> Rig:
        llm_ = FakeLLM([reply("Okay.", service_offered="no"), reply(REDIRECT), reply(REASON.question)])
        rig = rig_for(clinic(), llm_)
        rig.pacer.start()
        await rig.say("بدي أربعة كيلو بندورة")
        await rig.say("وكمان خيار")  # still not about teeth, and this time the model says nothing
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines() == [REDIRECT, closing_text(clinic(), "not_fit")]
    assert len(rig.of("scope_check")) == 1 and rig.of("loop_guard")[-1]["action"] == "off_topic"


def test_after_the_redirect_a_caller_who_does_want_us_carries_on() -> None:
    async def scenario(answer: str, third: Any) -> Rig:
        llm_ = FakeLLM([reply(REASON.question), reply(REASON.question), reply(REDIRECT), third, reply(STATUS.question)])
        rig = rig_for(clinic(), llm_)
        rig.pacer.start()
        await rig.say("الو")
        await rig.say("يعني هيك")
        await rig.say(answer)
        await rig.agent.close()
        return rig

    # a plain "yes" answers "is that what you need?", even if the model writes nothing down
    yes = run(lambda: scenario("آه", reply(REASON.question)))
    assert yes.agent.slots["service_offered"] == "yes" and yes.bot_lines()[-1] == REASON.question and yes.hangups == []
    need = run(lambda: scenario("عندي وجع أسنان", reply(STATUS.question, service_offered="yes", reason="وجع أسنان")))
    assert need.agent.slots["reason"] == "وجع أسنان" and need.bot_lines()[-1] == STATUS.question and need.hangups == []
    assert agrees("آه، بدي موعد", "ar") and agrees("כן בבקשה", "he") and agrees("Yes please") and not agrees("بدي بندورة", "ar") and not agrees("No.")


def test_without_a_scope_slot_nothing_learned_ends_in_a_call_back_not_the_next_question() -> None:
    async def scenario() -> Rig:
        llm_ = FakeLLM([reply(STATUS.question), reply(STATUS.question), reply("Sorry? First visit, or been here before?"), reply(STATUS.question)])
        rig = rig_for(clinic("en", slots=(STATUS, NAME)), llm_)
        rig.pacer.start()
        for text in ("Hello.", "Well.", "Hmm."):
            await rig.say(text)
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines()[-1] == closing_text(clinic("en"), "callback") and rig.hangups == ["needs-callback"]
    assert [g["action"] for g in rig.of("loop_guard")] == ["rephrase", "callback"] and rig.agent.slots["patient_status"] is None


# ====================================================================== loop guard keeps a good reply
def test_second_ask_keeps_the_models_own_new_wording_and_saves_a_request() -> None:
    own = "Sorry, I didn't get that. Have you been treated here before?"

    async def scenario() -> Rig:
        llm_ = FakeLLM([reply(STATUS.question, service_offered="yes"), reply(own), reply(STATUS.question), reply("Okay. Your name?")])
        rig = rig_for(clinic("en", slots=(SCOPE, STATUS, NAME)), llm_)
        rig.pacer.start()
        await rig.say("Hello, an appointment please.")
        await rig.say("Mmm.")
        assert len(llm_.requests) == 2 and rig.of("state")[-1]["asked"] == {"patient_status": 2}  # kept, and it counts as the second ask
        await rig.say("Well.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines() == [STATUS.question, own, "Okay. Your name?"]
    assert [g["action"] for g in rig.of("loop_guard")] == ["reworded", "skip"]  # something is known: going on is fine here


def test_the_very_same_sentence_is_never_said_twice_in_a_row() -> None:
    """702: the forced rephrase came back word for word as the question just asked."""

    async def scenario() -> Rig:
        llm_ = FakeLLM([reply(STATUS.question, service_offered="yes"), reply(STATUS.question), reply(STATUS.question)])
        rig = rig_for(clinic("en", slots=(SCOPE, STATUS, NAME)), llm_)
        rig.pacer.start()
        await rig.say("Hello, an appointment please.")
        await rig.say("Mmm.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines() == [STATUS.question, ENGLISH.didnt_catch]
    assert rig.of("loop_guard")[0]["action"] == "rephrase"


# ====================================================================== phone: refusal, fragments, read-back
@pytest.mark.parametrize(
    ("language", "said"),
    [("he", "מה צריך קול? לא חשוב המספר. תמשיך הלאה."), ("ar", "مش مهم الرقم، كمّل"), ("en", "Never mind the number, go on.")],
)
def test_real_call_701_never_mind_the_number_stops_the_asking(language: str, said: str) -> None:
    async def scenario() -> Rig:
        llm_ = FakeLLM([reply("What is the number?"), reply("Thank you, noted.")])
        rig = rig_for(clinic(language, slots=(SCOPE, STATUS, NAME, PHONE)), llm_)
        rig.agent.slots.update({"service_offered": "yes", "patient_status": "new", "name": "Dana"})
        rig.pacer.start()
        await rig.say(said)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.agent.slots["phone"] == UNCLEAR and rig.bot_lines()[0].startswith("Thank you, noted.")
    assert [(g["action"], g["slot"]) for g in rig.of("loop_guard")] == [("declined", "phone")]
    assert rig.agent.summary()["valid_phone"] is False and rig.agent.summary()["unclear_slots"] == ["phone"]


def test_words_that_do_and_do_not_mean_never_mind() -> None:
    assert skips_question("לא חשוב", "he") and skips_question("עזוב, לא רוצה לתת", "he") and skips_question("بلاش", "ar") and skips_question("مش ضروري", "ar")
    assert skips_question("I'd rather not") and skips_question("skip it")
    assert not skips_question("0521234567", "he") and not skips_question("מחר בשעה ארבע", "he")
    assert not skips_question("خلص شكرا", "ar") and not skips_question("My number is 0521234567")


def test_a_slot_that_decides_the_outcome_cannot_be_waved_away() -> None:
    async def scenario() -> Rig:
        llm_ = FakeLLM([reply("Do you own the house, or do you rent?")])
        rig = Rig(llm_, loop_guard=True)  # PROFILE: owns_home can disqualify
        rig.pacer.start()
        await rig.say("Never mind, go on.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.agent.slots["owns_home"] is None and rig.of("loop_guard") == []


def test_real_call_701_read_back_is_worded_by_the_program_in_each_language() -> None:
    """The model had said "שמעת" (you heard) for "שמעתי" (I heard)."""
    assert HEBREW.read_digits("0554412345") == "אפס חמש חמש, ארבע ארבע אחת, שתיים שלוש ארבע חמש"
    assert "{digits}" not in HEBREW.phone_incomplete and HEBREW.phone_incomplete.startswith("לא שמעתי את כל המספר.")
    assert ARABIC.phone_confirm.format(digits=ARABIC.read_digits("05")) == "سمعت صفر خمسة. مزبوط؟"
    assert ARABIC.read_digits("2") == "اتنين"  # not "تنين", which the voice may read as "dragon"
    for text in (ARABIC.phone_confirm, ARABIC.phone_incomplete, *ARABIC.digits):
        assert "ق" not in text  # the voice reads qaf the formal way (pronunciation test)

    async def scenario() -> Rig:
        llm_ = FakeLLM([reply("תודה, קיבלתי את הפרטים.", phone="05544")])
        rig = rig_for(clinic("he", slots=(SCOPE, STATUS, NAME, PHONE)), llm_)
        rig.agent.slots.update({"service_offered": "yes", "patient_status": "new", "name": "דנה"})
        rig.pacer.start()
        await rig.say("05544")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines() == ["לא שמעתי את כל המספר. אפשר לומר אותו שוב, לאט?"]
    assert len(rig.llm.requests) == 1 and rig.agent.slots["phone"] is None


# ====================================================================== scope wording in the profiles
@pytest.mark.parametrize("name", ["clinic", "clinic_he", "clinic_ar"])
def test_clinic_profiles_count_unrelated_requests_as_out_of_scope(name: str) -> None:
    profile = load_profile(name, PROFILES_DIR)
    scope = next(slot for slot in profile.slots if slot.name == "service_offered")
    assert "nothing to do with a clinic" in " ".join((scope.description or "").split())
    assert scope.question is None and "general chat" in " ".join(build_system_prompt(profile).split())


# ====================================================================== Groq limits
def test_fallback_reply_says_whether_the_minute_or_the_day_ran_out() -> None:
    def handler_for(text: str) -> Any:
        return lambda request: limited(text) if json.loads(request.content)["model"] == BIG else ok()

    minute, _ = ask(llm(handler_for(TPM)))
    day, _ = ask(llm(handler_for(TPD)))
    assert (minute["model"], minute["model_reason"]) == (SMALL, "minute") and (day["model"], day["model_reason"]) == (SMALL, "daily")
    record = {"type": "llm_fallback_model", "turn": 3, "model": SMALL, "reason": "minute"}
    assert format_record(record) == f"[llm_fallback_model] the main model's per-minute token limit was reached; {SMALL} answered this request"
    assert "daily token limit is used up" in (format_record({**record, "reason": "daily"}) or "")


def test_call_logs_each_cause_once() -> None:
    async def scenario() -> Rig:
        minute, day = {"model": SMALL, "model_reason": "minute"}, {"model": SMALL, "model_reason": "daily"}
        rig = Rig(FakeLLM([{**reply("Do you own the house?"), **minute}, {**reply("Do you own it?"), **minute}, {**reply("Thanks.", owns_home="yes"), **day}]))
        rig.pacer.start()
        for text in ("Hi.", "What?", "I own it."):
            await rig.say(text)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert [(r["turn"], r["reason"]) for r in rig.of("llm_fallback_model")] == [(1, "minute"), (3, "daily")]


def test_our_token_count_is_raised_to_groqs_own_figure() -> None:
    """2026-11-12: we showed 41k of 200k used while Groq refused the model for the day."""
    assert tokens_used(TPD) == 199211 and tokens_used("no figure here") is None
    ledger = TokenLedger()
    ledger.add(BIG, 41_000)
    assert ledger.correct(BIG, 199_211) == 158_211
    assert ledger.last_24h()[BIG] == {"tokens": 199_211, "requests": 1}  # tokens corrected, requests untouched
    assert ledger.correct(BIG, 150_000) == 0 and ledger.last_24h()[BIG]["tokens"] == 199_211  # never lowered

    def handler(request: httpx.Request) -> httpx.Response:
        return limited(TPD) if json.loads(request.content)["model"] == BIG else ok(tokens=1000)

    client = llm(handler, ledger=ledger)
    ask(client)
    assert client.ledger.last_24h()[BIG]["tokens"] == 199_211  # already matches: nothing added twice


# ====================================================================== no phone number: its own outcome
@pytest.mark.parametrize(
    ("name", "said", "closing"),
    [
        ("clinic_ar", "مش مهم الرقم، كمّل", "تمام، سجّلت طلبك. بس ما عنا نمرة إلك، فإذا بتحب، اتصل فينا مرة تانية لنأكّد الموعد. مع السلامة!"),
        ("clinic_he", "לא חשוב המספר, תמשיך הלאה", "רשמתי את הבקשה. מכיוון שלא השארתם מספר, אפשר להתקשר אלינו שוב כדי לאשר את התור. יום טוב!"),
        ("clinic", "Never mind the number, go on.", "I've noted your request. Since we don't have a number for you, please call us again to confirm the appointment. Take care!"),
    ],
)
def test_real_call_701_a_caller_who_gives_no_number_is_not_promised_a_call(name: str, said: str, closing: str) -> None:
    profile = load_profile(name, PROFILES_DIR)
    goodbye = {"clinic_ar": "لا شكرا", "clinic_he": "לא תודה", "clinic": "No, thank you."}[name]

    async def scenario() -> Rig:
        llm_ = FakeLLM([reply(next(s.question for s in profile.slots if s.name == "phone") or "")])
        rig = rig_for(profile, llm_)
        rig.agent.slots.update({"service_offered": "yes", "reason": "check-up", "patient_status": "new", "preferred_time": "Tuesday", "name": "Dana"})
        rig.pacer.start()
        await rig.say(said)
        await rig.say(goodbye)
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    summary = rig.agent.summary()
    assert summary["outcome"] == "qualified_no_callback" and summary["valid_phone"] is False and rig.agent.slots["phone"] == UNCLEAR
    assert rig.bot_lines()[-1] == closing and closing in fixed_phrases(profile)  # pre-generated with the other fixed phrases
    assert len(rig.llm.requests) == 1  # the announcement is the profile's sentence, not the model's
    spoken = " ".join(rig.bot_lines())
    assert profile.closing_qualified not in spoken and profile.confirmation.split("[")[0] in spoken
    for promise in ("رح نتصل", "תחזור אליכם", "will call you"):
        assert promise not in spoken


def test_the_no_number_outcome_does_not_make_the_phone_a_must_have() -> None:
    from app.slots import affects_outcome

    profile = load_profile("clinic_he", PROFILES_DIR)
    phone = next(s for s in profile.slots if s.name == "phone")
    assert not affects_outcome(phone, profile.outcomes)  # so "never mind" and the loop guard still let it go
    lead = load_profile("lead_qualification", PROFILES_DIR)
    assert affects_outcome(next(s for s in lead.slots if s.name == "callback_ok"), lead.outcomes)
