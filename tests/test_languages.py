"""Hebrew and Arabic: the program's own phrases, decline / goodbye words, prompt rules,
the two clinic profiles, mixed-script amounts and one profile per extension."""

from __future__ import annotations

import asyncio
import dataclasses
import socket
from pathlib import Path
from typing import Any

import pytest

from app.agent import APOLOGY_TEXT, ONE_MOMENT_TEXT, asks_anything_else, closing_text, fixed_phrases, goodbye_text
from app.amounts import parse_amount
from app.config import load_settings
from app.intent import caller_reply_kind, interrupts_closing
from app.lang import ARABIC, ENGLISH, HEBREW, pack
from app.profiles import ProfileError, load_profile, parse_profile
from app.prompt_builder import ANYTHING_ELSE_QUESTION, anything_else_question, build_system_prompt, incomplete_closing
from app.providers import check_providers, describe_stt, make_tts
from app.server import build_all_services, build_services, serve
from app.text_utils import clean_fallback_text
from app.tts_azure import AzureTTS
from tests.helpers import PROFILE, FakeLLM, FakePhrases, Rig, reply, run, until

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
ENV = {"DEEPGRAM_API_KEY": "dg", "GROQ_API_KEY": "groq", "AZURE_SPEECH_KEY": "az", "AZURE_SPEECH_REGION": "northeurope"}


def settings(tmp_path: Path, **extra: str) -> Any:
    s = load_settings(tmp_path / "none.env", environ={**ENV, "USAGE_DIR": str(tmp_path / "usage"), **extra})
    return dataclasses.replace(s, profiles_dir=PROFILES_DIR, tts_cache_dir=tmp_path / "tts")


# ====================================================================== language packs
def test_english_pack_is_what_the_program_always_said() -> None:
    assert pack("en") is ENGLISH and pack("en-US") is ENGLISH and pack("fr") is ENGLISH  # unknown: English
    assert APOLOGY_TEXT == "Sorry, I'm having a little trouble on my end. Could you say that again?"
    assert ONE_MOMENT_TEXT == "One moment, please." and ANYTHING_ELSE_QUESTION == "Is there anything else I can help you with?"
    assert incomplete_closing(PROFILE) == "Thank you for calling Acme Solar. Goodbye!"
    assert goodbye_text(PROFILE).startswith("I'm sorry, but we've reached the time limit")


@pytest.mark.parametrize("language, expected", [("he", HEBREW), ("ar", ARABIC), ("ar-JO", ARABIC), ("he-IL", HEBREW)])
def test_each_language_has_all_of_the_programs_phrases(language: str, expected: Any) -> None:
    p = pack(language)
    assert p is expected
    for text in (p.apology, p.one_moment, p.anything_else, p.neutral_closing, p.time_limit):
        assert text and clean_fallback_text(text.format(business=""), language) is not None  # written in that alphabet
    assert "{business}" in p.neutral_closing and "{business}" in p.time_limit and p.speaking_rule
    assert p.asks_anything_else(p.anything_else)


def test_arabic_program_phrases_avoid_the_letter_qaf() -> None:
    """Until the pronunciation of ق is settled, the lines the program itself speaks do without it."""
    for text in (ARABIC.apology, ARABIC.one_moment, ARABIC.anything_else, ARABIC.neutral_closing, ARABIC.time_limit):
        assert "ق" not in text


@pytest.mark.parametrize(
    "language, text, expected",
    [
        ("he", "אפשר לעזור בעוד משהו?", True), ("he", "יש עוד משהו שאפשר לעזור בו?", True), ("he", "באיזה יום נוח לכם?", False),
        ("ar", "في إشي تاني ممكن أساعدك فيه؟", True), ("ar", "بدك شي تاني؟", True), ("ar", "أي يوم بناسبك؟", False),
        ("en", "Anything else?", True), ("en", "Which day suits you?", False),
    ],
)
def test_anything_else_is_recognised_per_language(language: str, text: str, expected: bool) -> None:
    assert asks_anything_else(text, language) is expected


# ====================================================================== decline / goodbye words
@pytest.mark.parametrize(
    "language, text, kind",
    [
        ("he", "להתראות", "goodbye"), ("he", "תודה רבה, יום טוב!", "goodbye"), ("he", "ביי", "goodbye"),
        ("he", "לא תודה, זה הכול.", "decline"), ("he", "לא.", "decline"), ("he", "תודה רבה", "decline"), ("he", "בסדר, תודה", "decline"),
        ("he", "אין צורך", "decline"), ("he", "זהו", "decline"),
        ("he", "לא, אני גר בדירה שכורה ולא בבית", None), ("he", "כן", None), ("he", "יום שלישי בבוקר", None),
        ("he", "שלום, אני רוצה לקבוע תור", None), ("he", "לא תודה, אבל מה שעות הפתיחה?", None), ("he", "", None),
        ("ar", "مع السلامة", "goodbye"), ("ar", "يلا باي", "goodbye"), ("ar", "الله معك", "goodbye"),
        ("ar", "لا شكراً، بس هيك.", "decline"), ("ar", "لا", "decline"), ("ar", "لأ", "decline"), ("ar", "شكراً كتير", "decline"),
        ("ar", "تمام، شكرا", "decline"), ("ar", "خلص، يعطيك العافية", "decline"), ("ar", "ما في شي تاني", "decline"),
        ("ar", "اه", None), ("ar", "يوم الثلاثا الصبح", None), ("ar", "مرحبا بدي احجز موعد", None),
        ("ar", "السلام عليكم", None), ("ar", "لا شكرا، بس شو ساعات الدوام؟", None),
        ("ar-JO", "مع السلامه", "goodbye"),
    ],
)
def test_decline_and_goodbye_words(language: str, text: str, kind: Any) -> None:
    assert caller_reply_kind(text, language) == kind


@pytest.mark.parametrize(
    "language, text, expected",
    [
        ("he", "רגע, עוד שאלה", True), ("he", "מה שעות הפתיחה?", True), ("he", "אל תנתקי", True), ("he", "בעצם יש לי שאלה", True),
        ("he", "להתראות", False), ("he", "תודה", False), ("he", "אוקיי", False), ("he", "בוא נלך לאכול", False),
        ("ar", "لحظة، عندي سؤال", True), ("ar", "شو ساعات الدوام؟", True), ("ar", "استنى", True), ("ar", "لا تسكر", True),
        ("ar", "مع السلامة", False), ("ar", "شكراً", False), ("ar", "تمام", False), ("ar", "يلا نروح ناكل", False),
    ],
)
def test_what_may_interrupt_the_closing_line(language: str, text: str, expected: bool) -> None:
    assert interrupts_closing(text, language) is expected


def test_english_rules_are_untouched_by_the_language_argument() -> None:
    assert caller_reply_kind("No thanks.") == caller_reply_kind("No thanks.", "en") == "decline"
    assert caller_reply_kind("לא תודה") is None  # Hebrew words mean nothing to the English rules
    assert interrupts_closing("Wait.") is True and interrupts_closing("Goodbye.", "en-US") is False


# ====================================================================== amounts written half in words, half in digits
@pytest.mark.parametrize(
    "phrase, amount, currency",
    [
        ("ميتين و50 شيكل", 250, "ILS"),  # seen live from speech-to-text
        ("מאתיים ו-50 שקלים", 250, "ILS"),
        ("two hundred and 50 dollars", 250, "USD"),
        ("ألف و500", 1500, None),
        ("خمسمية و 20", 520, None),
        ("مية و5", 105, None),
    ],
)
def test_mixed_word_and_digit_amounts(phrase: str, amount: int, currency: Any) -> None:
    assert parse_amount(phrase) == (amount, currency)


@pytest.mark.parametrize("phrase", ["between 400 and 500", "400 and 500 shekels", "500 and five", "150 or 200"])
def test_two_separate_numbers_are_still_not_guessed(phrase: str) -> None:
    assert parse_amount(phrase)[0] is None


# ====================================================================== prompt
def test_english_prompt_has_no_language_section() -> None:
    assert "LANGUAGE" not in build_system_prompt(PROFILE)


@pytest.mark.parametrize("name, language, dialect", [("clinic_he", "Hebrew", "spoken Israeli Hebrew"), ("clinic_ar", "Arabic", "spoken Levantine Arabic")])
def test_prompt_tells_the_model_to_speak_only_the_profile_language(name: str, language: str, dialect: str) -> None:
    prompt = build_system_prompt(load_profile(name, PROFILES_DIR))
    assert f'Every "say" is in {language} only' in prompt and dialect in prompt
    assert "words, not digits" in prompt  # the voices have no reliable number markup
    assert "stay exactly as listed, in English" in prompt  # slot keywords are not translated
    assert prompt.index("LANGUAGE") < prompt.index("HOW TO SPEAK") and "1 or 2 short spoken sentences" in prompt
    assert prompt == build_system_prompt(load_profile(name, PROFILES_DIR))  # still static: cacheable


def test_speech_notes_are_added_to_the_language_rules() -> None:
    profile = dataclasses.replace(load_profile("clinic_ar", PROFILES_DIR), speech_notes="Prefer words without the letter qaf.")
    prompt = build_system_prompt(profile)
    assert "- Prefer words without the letter qaf." in prompt
    chosen = load_profile("clinic_ar", PROFILES_DIR).speech_notes  # the rule chosen after the pronunciation test
    assert "white dialect" in chosen and "نمرة تلفونك instead of رقم تليفونك" in chosen
    assert "not absolute: keep a word with ق only when there is no natural alternative" in chosen and len(chosen) <= 600
    assert f"- {chosen}" in build_system_prompt(load_profile("clinic_ar", PROFILES_DIR))
    with pytest.raises(ProfileError, match="speech_notes"):
        parse_profile("t", {"speech_notes": 5})


# ====================================================================== the two clinic profiles
@pytest.mark.parametrize("name", ["clinic_he", "clinic_ar"])
def test_clinic_profiles_have_the_same_slots_as_the_english_one(name: str) -> None:
    english, other = load_profile("clinic", PROFILES_DIR), load_profile(name, PROFILES_DIR)

    def shape(profile: Any) -> list[Any]:
        return [(s.name, s.type, s.required, s.values, s.disqualifying, bool(s.question)) for s in profile.slots]

    assert shape(other) == shape(english)
    language = other.language
    for text in [other.greeting, other.closing_qualified, other.closing_not_fit, *[s.question for s in other.slots if s.question]]:
        assert clean_fallback_text(text, language) is not None, text  # written in the profile's language
    assert other.closing_qualified != other.closing_not_fit and other.keyterms


def test_hebrew_profile_wiring(tmp_path: Path) -> None:
    profile = load_profile("clinic_he", PROFILES_DIR)
    s = settings(tmp_path)
    check_providers(profile, s)
    assert profile.language == "he" and describe_stt(profile, s) == "nova-3/he" and profile.tts.voice == "he-IL-HilaNeural"
    assert caller_reply_kind("לא תודה", profile.language) == "decline"


def test_arabic_profile_wiring_and_no_qaf_in_fixed_lines(tmp_path: Path) -> None:
    profile = load_profile("clinic_ar", PROFILES_DIR)
    s = settings(tmp_path)
    check_providers(profile, s)
    assert describe_stt(profile, s) == "nova-3/multi[ar,he]" and profile.tts.voice == "ar-JO-SanaNeural"
    spoken_as_is = fixed_phrases(profile) + [slot.question for slot in profile.slots if slot.question]
    assert [text for text in spoken_as_is if "ق" in text] == []


def test_fixed_phrases_are_in_the_profile_language() -> None:
    hebrew = load_profile("clinic_he", PROFILES_DIR)
    phrases = fixed_phrases(hebrew)
    assert phrases[0] == hebrew.greeting and HEBREW.apology in phrases and HEBREW.one_moment in phrases and HEBREW.anything_else in phrases
    assert incomplete_closing(hebrew) == "תודה שהתקשרתם למרפאת השיניים הרבור. להתראות!" and incomplete_closing(hebrew) in phrases
    assert goodbye_text(hebrew) in phrases and "מרפאת השיניים הרבור" in goodbye_text(hebrew)
    assert anything_else_question(hebrew) == HEBREW.anything_else
    assert APOLOGY_TEXT not in phrases and len(set(phrases)) == len(phrases) == 8
    assert closing_text(hebrew, "incomplete") == incomplete_closing(hebrew)


# ====================================================================== a Hebrew call end to end (fake model and audio)
def hebrew_rig(llm: FakeLLM, **kwargs: Any) -> Rig:
    profile = dataclasses.replace(PROFILE, language="he", greeting="שלום.", closing_qualified="תודה, נחזור אליכם. יום טוב!", closing_not_fit="מצטערת, לא נוכל לעזור. יום טוב!")
    return Rig(llm, profile=profile, phrases=FakePhrases(known=set(fixed_phrases(profile))), **kwargs)


def test_hebrew_call_uses_hebrew_for_everything_the_program_says() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM([reply("תודה.", owns_home="no"), reply("אנחנו מתקינים רק בבתים פרטיים. אפשר לעזור בעוד משהו?")])
        rig = hebrew_rig(llm, end_silence_s=0.05)
        rig.pacer.start()
        await rig.say("אני גר בדירה שכורה.")
        await rig.say("לא תודה, זה הכול.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    # the model's own "anything else?" was dropped; the server asked it once, in Hebrew
    assert rig.bot_lines() == [f"אנחנו מתקינים רק בבתים פרטיים. {HEBREW.anything_else}", "מצטערת, לא נוכל לעזור. יום טוב!"]
    assert rig.hangups == ["not-a-fit"] and len(rig.llm.requests) == 2  # the Hebrew decline needed no model
    assert HEBREW.anything_else in rig.phrases.served and ANYTHING_ELSE_QUESTION not in rig.phrases.served


def test_hebrew_goodbye_and_apology() -> None:
    async def scenario() -> Rig:
        from app.llm_groq import LLMError

        rig = hebrew_rig(FakeLLM([LLMError("Groq HTTP 500"), reply("Okay.")]), end_silence_s=0.05)
        rig.pacer.start()
        await rig.say("שלום, יש לי שאלה.")
        await rig.say("טוב, להתראות.")
        await until(lambda: bool(rig.hangups))
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.of("bot")[0] == {"type": "bot", "turn": None, "text": HEBREW.apology, "kind": "apology"}
    assert rig.bot_lines()[-1] == "תודה שהתקשרתם לAcme Solar. להתראות!" and rig.hangups == ["caller-goodbye"]


def test_hebrew_remark_during_the_closing_is_ignored_and_a_question_is_not() -> None:
    async def scenario(words: str) -> Rig:
        llm = FakeLLM([reply("תודה.", owns_home="no"), reply("רק בבתים פרטיים."), reply("הביקור חינם.")])
        rig = hebrew_rig(llm, end_silence_s=0.1)
        rig.pacer.start()
        await rig.say("אני גר בדירה שכורה.")
        rig.event("StartOfTurn", "תודה")
        rig.event("EndOfTurn", "תודה רבה.")
        await until(lambda: len(rig.of("ending")) == 1)
        rig.event("StartOfTurn", words.split()[0])
        rig.event("EndOfTurn", words)
        await asyncio.sleep(0.4)
        await rig.agent.close()
        return rig

    assert run(lambda: scenario("להתראות")).hangups == ["not-a-fit"]
    asked = run(lambda: scenario("רגע, הביקור חינם?"))
    assert asked.hangups == [] and len(asked.of("hangup_cancelled")) == 1


# ====================================================================== one profile per extension
def free_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for s in sockets:
            s.bind(("127.0.0.1", 0))
        return [s.getsockname()[1] for s in sockets]
    finally:
        for s in sockets:
            s.close()


def test_extra_profiles_setting() -> None:
    s = load_settings(Path("/nonexistent.env"), environ={"EXTRA_PROFILES": "clinic_he:9093, clinic_ar:9094"})
    assert s.extra_profiles == (("clinic_he", 9093), ("clinic_ar", 9094))
    assert load_settings(Path("/nonexistent.env"), environ={}).extra_profiles == ()
    for bad in ("clinic_he", "clinic_he:abc", "clinic_he:80", "a:9093,b:9093"):
        with pytest.raises(ValueError, match="EXTRA_PROFILES"):
            load_settings(Path("/nonexistent.env"), environ={"EXTRA_PROFILES": bad})


def test_each_extra_profile_becomes_its_own_line_sharing_connections_and_limits(tmp_path: Path) -> None:
    async def scenario() -> None:
        s = settings(tmp_path, EXTRA_PROFILES="clinic_he:9093,clinic_ar:9094,missing_profile:9095")
        lines = build_all_services(s)
        assert [(line.profile.name, line.port) for line in lines] == [("lead_qualification", None), ("clinic_he", 9093), ("clinic_ar", 9094)]
        english, hebrew, arabic = lines  # the missing profile was logged and skipped, not fatal
        assert hebrew.http is english.http and arabic.http is english.http
        assert hebrew.llm.budget is english.llm.budget and hebrew.llm is not english.llm
        assert (english.llm.fallback_language, hebrew.llm.fallback_language, arabic.llm.fallback_language) == ("en", "he", "ar")
        assert isinstance(hebrew.tts, AzureTTS) and isinstance(arabic.tts, AzureTTS)
        assert hebrew.tts.voice == "he-IL-HilaNeural" and arabic.tts.voice == "ar-JO-SanaNeural"
        assert hebrew.tts.limiter is arabic.tts.limiter and hebrew.tts.usage is arabic.tts.usage  # one Azure subscription, one limit
        assert english.tts.keep_warm_url and hebrew.tts.keep_warm_url is None
        assert hebrew.system_prompt != english.system_prompt and "Hebrew only" in hebrew.system_prompt
        await english.http.aclose()

    asyncio.run(scenario())


def test_an_extra_profile_on_the_main_port_is_refused(tmp_path: Path) -> None:
    async def scenario() -> int:
        lines = build_all_services(settings(tmp_path, EXTRA_PROFILES="clinic_he:9092"))
        await lines[0].http.aclose()
        return len(lines)

    assert asyncio.run(scenario()) == 1


def test_each_line_listens_on_its_own_port(tmp_path: Path) -> None:
    async def scenario() -> list[bool]:
        main_port, extra_port = free_ports(2)
        s = dataclasses.replace(settings(tmp_path), port=main_port, agent_enabled=False)  # no network: plain listeners
        lines = [build_services(s), build_services(s, port=extra_port)]
        task = asyncio.get_running_loop().create_task(serve(lines))
        await asyncio.sleep(0.1)
        reachable = []
        for port in (main_port, extra_port):
            try:
                _, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.close()
                reachable.append(True)
            except OSError:
                reachable.append(False)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return reachable

    assert asyncio.run(scenario()) == [True, True]


def test_arabic_wording_approved_on_2026_11_12() -> None:
    arabic = load_profile("clinic_ar", PROFILES_DIR)
    phrases = fixed_phrases(arabic)
    assert ARABIC.anything_else == "في إشي تاني ممكن أساعدك فيه؟" and ARABIC.asks_anything_else(ARABIC.anything_else)
    assert incomplete_closing(arabic) == "شكراً لاتصالك بعيادة الشفاء لطب الأسنان. مع السلامة!"
    assert goodbye_text(arabic) == "آسفة، لازم ننهي المكالمة هلأ. شكراً لاتصالك بعيادة الشفاء لطب الأسنان. مع السلامة!"
    assert not any("ـ" in text for text in phrases)  # no tatweel anywhere
    questions = {slot.name: slot.question for slot in arabic.slots}
    assert questions["preferred_time"] == "أي يوم وأي ساعة بيناسبوك؟"
    assert questions["phone"] == "على أي نمرة نتصل فيك لنأكّد الموعد؟"
