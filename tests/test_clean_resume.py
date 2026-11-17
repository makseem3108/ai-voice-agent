"""Real calls of 17 Nov (701 and 702): the bot's voice "broke" around the phone number.

Cause: the read-back of a number took 15 seconds, so the caller answered before
it ended; the bot paused on the sound, speech-to-text did not confirm in time,
and the bot carried on from the middle of a word - then paused again 130 ms later.

Four fixes: the number is read in groups (050, 123, 4567); an incomplete number
is not read back at all (one short fixed request); after a false pause the bot
never continues mid-word (a short rest is dropped, a longer one starts the
sentence again); and for a second after that, sound alone cannot pause it again."""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from app.agent import DETECTOR_REST_S, fixed_phrases
from app.audio_out import FRAME_BYTES, Pacer
from app.call_log import format_record
from app.lang import ARABIC, ENGLISH, HEBREW
from app.phones import digit_groups
from app.slots import SlotDef
from tests.helpers import FRAME, NAME, PROFILE, FakeLLM, FakeTTS, Rig, run, until
from tests.test_agent import FakeTime
from tests.test_fixes import FakeDetector

PHONE = SlotDef(name="phone", type="phone", question="What is the best phone number?")


# ---------------------------------------------------------------- 1. groups
@pytest.mark.parametrize(
    "number, groups",
    [
        ("0501234567", ["050", "123", "4567"]),  # mobile: 3-3-4
        ("049991234", ["04", "999", "1234"]),  # landline: 2-3-4
        ("0771234567", ["077", "123", "4567"]),
        ("050 123-4567", ["050", "123", "4567"]),
        ("05544", ["055", "44"]),  # not a whole number: threes
        ("", []),
    ],
)
def test_numbers_are_grouped_the_way_people_say_them(number: str, groups: list[str]) -> None:
    assert digit_groups(number) == groups


def test_read_back_pauses_between_groups_only_and_keeps_the_digit_words() -> None:
    assert ENGLISH.read_digits("0501234567") == "zero five zero, one two three, four five six seven"
    assert HEBREW.read_digits("0501234567") == "אפס חמש אפס, אחת שתיים שלוש, ארבע חמש שש שבע"
    assert ARABIC.read_digits("0501234567") == "صفر خمسة صفر, واحد اتنين تلاتة, أربعة خمسة ستة سبعة"
    assert HEBREW.read_digits("049991234") == "אפס ארבע, תשע תשע תשע, אחת שתיים שלוש ארבע"
    for language in (ENGLISH, HEBREW, ARABIC):
        assert language.read_digits("0501234567").count(",") == 2  # was 9 pauses, one after every digit


# ---------------------------------------------------------------- 2. incomplete number
def test_incomplete_number_gets_the_short_fixed_request_without_digits() -> None:
    for language in (ENGLISH, HEBREW, ARABIC):
        assert "{" not in language.phone_incomplete and not any(word in language.phone_incomplete for word in language.digits)
        assert len(language.phone_incomplete) < 70
    with_phone = dataclasses.replace(PROFILE, slots=(NAME, PHONE))
    assert ENGLISH.phone_incomplete in fixed_phrases(with_phone)  # pre-generated: no synthesis during the call
    assert ENGLISH.phone_incomplete not in fixed_phrases(dataclasses.replace(PROFILE, slots=(NAME,)))


def test_incomplete_number_is_logged_as_asked_again_not_read_back() -> None:
    line = format_record({"type": "phone_readback", "slot": "phone", "heard": "0 5 5 4 4", "complete": False}) or ""
    assert "asked for it again" in line and "to confirm it" not in line
    line = format_record({"type": "phone_readback", "slot": "phone", "heard": "0 5 5", "complete": True}) or ""
    assert line.endswith("back to the caller to confirm it")


# ---------------------------------------------------------------- 3. pacer: never continue mid-word
def numbered(first: int, count: int) -> bytes:
    return b"".join(bytes([first + i]) * FRAME_BYTES for i in range(count))


def paced(pause_after: int, sentences: list[bytes]) -> tuple[list[int], str]:
    """Play ``sentences``; pause after ``pause_after`` frames, resume cleanly. Returns the frames sent."""

    async def main() -> tuple[list[int], str]:
        fake = FakeTime()
        sent: list[int] = []
        pacer = Pacer(lambda frame: sent.append(frame[0]), clock=fake.clock, sleep=fake.sleep)

        def on_frame() -> None:
            if len(sent) == pause_after:
                pacer.pause()

        pacer._on_frame = on_frame
        pacer.start()
        for audio in sentences:
            pacer.push(audio[:1000], new_sentence=True)  # arrives in chunks, like a stream
            pacer.push(audio[1000:])
        pacer.flush()
        for _ in range(pause_after + 20):
            await asyncio.sleep(0)
        assert pacer.paused and len(sent) == pause_after
        action = pacer.resume_cleanly()
        await asyncio.wait_for(pacer.drained(), 1)
        await pacer.stop()
        return sent, action

    return asyncio.run(main())


def test_long_rest_restarts_the_sentence_from_its_beginning() -> None:
    sent, action = paced(30, [numbered(0, 100)])  # 2 s sentence, paused after 0.6 s: 1.4 s left
    assert action == "restarted"
    assert sent == list(range(30)) + list(range(100))


def test_short_rest_is_dropped() -> None:
    sent, action = paced(60, [numbered(0, 100)])  # 0.8 s left
    assert action == "dropped" and sent == list(range(60))


def test_only_the_interrupted_sentence_is_repeated_and_the_next_one_follows() -> None:
    first, second = numbered(0, 20), numbered(100, 100)
    sent, action = paced(50, [first, second])  # paused 0.6 s into the second sentence
    assert action == "restarted"
    assert sent == list(range(20)) + list(range(100, 130)) + list(range(100, 200))
    sent, action = paced(10, [first, second])  # 0.2 s left of the first: dropped, the second is said whole
    assert action == "dropped" and sent == list(range(10)) + list(range(100, 200))


def test_resume_cleanly_does_nothing_when_not_paused() -> None:
    async def main() -> str:
        return Pacer(lambda frame: None).resume_cleanly()

    assert asyncio.run(main()) == "running"


# ---------------------------------------------------------------- 3 + 4. in a call
async def false_pause(rig: Rig, detector: FakeDetector, after_frames: int) -> None:
    rig.pacer.start()
    rig.event("EndOfTurn", "Tell me.")
    await until(lambda: len(rig.frames) >= after_frames)
    detector.fire = True  # a sound on the caller's side that speech-to-text never confirms
    rig.agent.on_caller_audio(FRAME)
    assert rig.pacer.paused
    await until(lambda: not rig.pacer.paused)


def test_false_pause_in_a_long_sentence_says_it_again_and_logs_it() -> None:
    async def scenario() -> Rig:
        detector = FakeDetector()
        rig = Rig(FakeLLM(["A long answer."]), FakeTTS(frames_per_sentence=70), detector=detector, barge_confirm_s=0.05)
        await false_pause(rig, detector, 5)
        await until(lambda: rig.agent._job is None)
        await asyncio.wait_for(rig.pacer.drained(), 4)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    (resume,) = rig.of("barge_resume")
    assert resume["action"] == "restarted" and "said again from its beginning" in (format_record(resume) or "")
    assert 75 <= len(rig.frames) <= 80  # the few frames before the pause + the whole sentence again
    assert rig.agent.history[-1]["content"] == "A long answer." and rig.agent.false_pauses == 1


def test_after_a_false_pause_sound_alone_cannot_pause_again_for_a_second() -> None:
    async def scenario() -> tuple[Rig, FakeDetector, bool, bool]:
        detector = FakeDetector()
        rig = Rig(FakeLLM(["A long answer."]), FakeTTS(frames_per_sentence=70), detector=detector, barge_confirm_s=0.05)
        await false_pause(rig, detector, 5)
        fed = detector.fed
        for _ in range(10):  # the same noise goes on (17 Nov: the bot stopped again after 130 ms)
            rig.agent.on_caller_audio(FRAME)
        ignored = not rig.pacer.paused and detector.fed == fed
        rig.agent._detector_off_until = 0.0  # ... a second later
        rig.agent.on_caller_audio(FRAME)
        paused_again = rig.pacer.paused
        await rig.agent.close()
        return rig, detector, ignored, paused_again

    rig, detector, ignored, paused_again = run(scenario)
    assert DETECTOR_REST_S == 1.0
    assert ignored and paused_again and len(rig.of("barge_pause")) == 2


def test_speech_to_text_can_still_interrupt_during_that_second() -> None:
    async def scenario() -> Rig:
        detector = FakeDetector()
        rig = Rig(FakeLLM(["A long answer."]), FakeTTS(frames_per_sentence=70), detector=detector, barge_confirm_s=0.05)
        await false_pause(rig, detector, 5)
        rig.event("StartOfTurn")  # the caller really speaks
        await asyncio.sleep(0.05)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.agent.barge_ins == 1 and len(rig.frames) < 20
