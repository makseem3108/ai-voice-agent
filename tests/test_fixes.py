"""Tests for the fixes made after the first real conversation:
fast local barge-in (pause -> confirm/resume), cancellable endings with two
outcomes, keyterms, diagnostic records and the debug audio recorder."""

from __future__ import annotations

import asyncio
import math
import os
import struct
import time
import wave
from pathlib import Path
from typing import Any, Optional

import pytest

from app.audio_out import FRAME_BYTES, Pacer
from app.call_log import format_record
from app.config import load_settings
from app.recording import InboundRecorder, cleanup_old_recordings
from app.stt_flux import FluxParams, TurnEvent, parse_flux_message
from app.vad import BargeInDetector, frame_rms
from tests.helpers import FRAME, FakeLLM, FakePhrases, FakeTTS, Rig, run, settle, until
from tests.test_agent import FakeTime, turn_info


def tone(amplitude: int) -> bytes:
    """One 20 ms frame of a 300 Hz tone at the given peak amplitude."""
    return struct.pack("<160h", *[int(amplitude * math.sin(2 * math.pi * 300 * i / 8000)) for i in range(160)])


# ====================================================================== pacer pause / resume
def test_pacer_pause_keeps_the_queue_and_resume_continues() -> None:
    async def main() -> tuple[int, int, bool, int]:
        fake = FakeTime()
        frames: list[bytes] = []
        pacer = Pacer(frames.append, clock=fake.clock, sleep=fake.sleep)
        pacer.start()
        audio = bytes(i % 251 for i in range(FRAME_BYTES * 10))
        pacer.push(audio)
        while len(frames) < 3:
            await asyncio.sleep(0)
        pacer.pause()
        await asyncio.sleep(0.01)
        at_pause = len(frames)
        await asyncio.sleep(0.02)
        still = len(frames)
        speaking_while_paused = pacer.speaking and pacer.paused
        pacer.resume()
        await asyncio.wait_for(pacer.drained(), 1)
        await pacer.stop()
        assert b"".join(frames) == audio  # nothing lost, nothing repeated
        return at_pause, still, speaking_while_paused, len(frames)

    at_pause, still, speaking_while_paused, total = asyncio.run(main())
    assert still == at_pause <= 4  # at most the frame already in flight
    assert speaking_while_paused is True  # "the bot has more to say" stays true while paused
    assert total == 10


def test_pacer_clear_while_paused_and_resume_is_idempotent() -> None:
    async def main() -> tuple[int, bool]:
        fake = FakeTime()
        frames: list[bytes] = []
        pacer = Pacer(frames.append, clock=fake.clock, sleep=fake.sleep)
        pacer.start()
        pacer.pause()
        pacer.push(FRAME * 5)
        await asyncio.sleep(0.01)
        assert frames == []  # paused before anything was sent
        dropped = pacer.clear()
        pacer.resume()
        pacer.resume()
        await asyncio.wait_for(pacer.drained(), 1)
        speaking = pacer.speaking
        await pacer.stop()
        return dropped, speaking

    assert asyncio.run(main()) == (5, False)


# ====================================================================== voice detector
def test_frame_rms() -> None:
    assert frame_rms(bytes(320)) == 0.0
    assert frame_rms(struct.pack("<4h", 1000, -1000, 1000, -1000)) == pytest.approx(1000.0)
    assert frame_rms(tone(2000)) == pytest.approx(2000 / math.sqrt(2), rel=0.02)
    assert frame_rms(b"") == 0.0


def test_detector_fires_after_enough_loud_voice_frames() -> None:
    detector = BargeInDetector(min_rms=500, min_speech_ms=120, window_ms=200, is_speech=lambda f: True)
    loud = tone(2000)
    results = [detector.feed(loud) for _ in range(6)]
    assert results == [False] * 5 + [True]  # 6 frames x 20 ms = 120 ms of speech


def test_detector_ignores_quiet_voice_like_echo_of_the_bot() -> None:
    # Worst case: the VAD says "voice" for every frame (echo of the bot IS voice),
    # but it is quiet. The loudness gate must keep it from interrupting.
    detector = BargeInDetector(min_rms=500, is_speech=lambda f: True)
    echo = tone(450)  # RMS ~320, like speech leaking back 20 dB down
    assert not any(detector.feed(echo) for _ in range(200))
    assert detector.last_rms < 500


def test_detector_ignores_loud_noise_that_is_not_voice() -> None:
    detector = BargeInDetector(min_rms=500, is_speech=lambda f: False)
    assert not any(detector.feed(tone(8000)) for _ in range(100))


def test_detector_needs_speech_within_the_window_not_scattered() -> None:
    detector = BargeInDetector(min_rms=500, min_speech_ms=120, window_ms=200, is_speech=lambda f: True)
    loud, quiet = tone(2000), bytes(320)
    fired = [detector.feed(loud if i % 3 == 0 else quiet) for i in range(60)]  # a click every 60 ms
    assert not any(fired)


def test_detector_reset_and_wrong_frame_size() -> None:
    detector = BargeInDetector(min_rms=500, min_speech_ms=60, is_speech=lambda f: True)
    loud = tone(2000)
    assert [detector.feed(loud) for _ in range(3)] == [False, False, True]
    detector.reset()
    assert detector.feed(loud) is False
    assert detector.feed(loud[:100]) is False  # not a 20 ms frame: ignored


def test_real_webrtc_vad_is_quiet_on_silence_and_handles_our_frame_size() -> None:
    detector = BargeInDetector()  # real webrtcvad, default thresholds
    assert not any(detector.feed(bytes(320)) for _ in range(100))
    detector.feed(tone(3000))  # must accept 20 ms @ 8 kHz without raising


# ====================================================================== agent: two-stage barge-in
class FakeDetector:
    """Fires when told to."""

    def __init__(self) -> None:
        self.fire = False
        self.last_rms = 1234.0
        self.fed = 0

    def feed(self, frame: bytes) -> bool:
        self.fed += 1
        return self.fire

    def reset(self) -> None:
        pass


def test_local_voice_pauses_the_bot_at_once_and_stt_confirms_the_barge_in() -> None:
    async def scenario() -> tuple[Rig, int, int]:
        detector = FakeDetector()
        rig = Rig(FakeLLM(["A long explanation."]), FakeTTS(frames_per_sentence=200), detector=detector, barge_confirm_s=2.0)
        rig.pacer.start()
        rig.event("EndOfTurn", "Tell me.")
        await until(lambda: len(rig.frames) >= 3)

        detector.fire = True  # the caller says "wait"
        rig.agent.on_caller_audio(FRAME)
        detector.fire = False
        frames_at_pause = len(rig.frames)
        await asyncio.sleep(0.12)  # Flux is slow; the bot must already be silent
        frames_before_confirm = len(rig.frames)
        assert rig.pacer.paused and rig.agent.barge_ins == 0

        rig.event("StartOfTurn", "Wait")  # now Flux confirms
        await asyncio.sleep(0.05)
        await rig.agent.close()
        return rig, frames_at_pause, frames_before_confirm

    rig, at_pause, before_confirm = run(scenario)
    assert before_confirm - at_pause <= 1  # silent from the moment of the local pause
    assert len(rig.frames) == before_confirm  # and nothing after the confirmation
    assert rig.kinds().count("barge_pause") == 1
    (barge,) = rig.of("barge_in")
    assert barge["local_pause_ms"] >= 100  # the pause beat Flux by this much
    assert barge["dropped_frames"] > 150
    assert rig.agent.barge_ins == 1 and rig.agent.false_pauses == 0
    assert rig.pacer.paused is False  # ready for the next reply
    assert "already paused" in (format_record(barge) or "")


def test_unconfirmed_local_pause_resumes_where_it_stopped() -> None:
    async def scenario() -> Rig:
        detector = FakeDetector()
        rig = Rig(FakeLLM(["Short answer."]), FakeTTS(frames_per_sentence=12), detector=detector, barge_confirm_s=0.1)
        rig.pacer.start()
        rig.event("EndOfTurn", "Tell me.")
        await until(lambda: len(rig.frames) >= 3)
        detector.fire = True  # a cough / echo, not a real interruption
        rig.agent.on_caller_audio(FRAME)
        detector.fire = False
        assert rig.pacer.paused
        await settle(rig)  # no StartOfTurn arrives -> the bot resumes by itself
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert len(rig.frames) == 12  # the whole answer was played after all
    assert rig.agent.false_pauses == 1 and rig.agent.barge_ins == 0
    (resume,) = rig.of("barge_resume")
    assert 80 <= resume["paused_ms"] <= 400
    assert rig.agent.history[-1]["content"] == "Short answer."  # the reply was not cancelled
    assert rig.llm.cancelled == 0


def test_detector_is_only_consulted_while_the_bot_is_speaking() -> None:
    async def scenario() -> tuple[Rig, FakeDetector]:
        detector = FakeDetector()
        detector.fire = True
        rig = Rig(detector=detector)
        rig.pacer.start()
        for _ in range(20):  # caller talks while the bot is silent: normal turn, no pause
            rig.agent.on_caller_audio(FRAME)
        await asyncio.sleep(0.01)
        await rig.agent.close()
        return rig, detector

    rig, detector = run(scenario)
    assert detector.fed == 0 and rig.kinds() == [] and not rig.pacer.paused


def test_end_of_turn_clears_a_pending_pause_so_the_reply_is_heard() -> None:
    async def scenario() -> Rig:
        detector = FakeDetector()
        rig = Rig(FakeLLM(["First.", "Second answer."]), FakeTTS(frames_per_sentence=100), detector=detector, barge_confirm_s=5.0)
        rig.pacer.start()
        rig.event("EndOfTurn", "One.")
        await until(lambda: len(rig.frames) >= 3)
        detector.fire = True
        rig.agent.on_caller_audio(FRAME)
        detector.fire = False
        rig.tts.frames = 3
        rig.event("StartOfTurn", "Two")
        rig.event("EndOfTurn", "Two.")
        await settle(rig)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.agent.history[-1]["content"] == "Second answer."
    assert rig.pacer.paused is False


def test_broken_detector_is_disabled_not_fatal() -> None:
    class Broken(FakeDetector):
        def feed(self, frame: bytes) -> bool:
            raise RuntimeError("vad exploded")

    async def scenario() -> Rig:
        rig = Rig(FakeLLM(["Answer."]), FakeTTS(frames_per_sentence=20), detector=Broken())
        rig.pacer.start()
        rig.event("EndOfTurn", "Hi.")
        await until(lambda: len(rig.frames) >= 2)
        rig.agent.on_caller_audio(FRAME)
        rig.agent.on_caller_audio(FRAME)
        await settle(rig)
        await rig.agent.close()
        return rig

    assert len(run(scenario).frames) == 20


# ====================================================================== agent: endings


















# ====================================================================== diagnostics


def test_every_flux_turn_event_is_recorded_with_audio_timestamps() -> None:
    async def scenario() -> Rig:
        rig = Rig()
        rig.pacer.start()
        now = time.time()
        rig.agent.on_stt_event(TurnEvent("StartOfTurn", 3, "Wa", 0.1, 4.0, 3.5, None), now)
        rig.agent.on_stt_event(TurnEvent("Update", 3, "Wait", 0.3, 4.4, 3.5, 4.3), now)
        await asyncio.sleep(0.01)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    start, update = rig.of("stt_event")
    assert start["event"] == "StartOfTurn" and start["turn_index"] == 3
    assert (start["audio_start"], start["audio_end"]) == (3.5, 4.0)
    assert start["bot_speaking"] is False and isinstance(start["lag_ms"], int)
    assert update["transcript"] == "Wait" and update["last_word_end"] == 4.3
    assert (format_record(start) or "").startswith("[stt] StartOfTurn")
    assert format_record(update) is None  # updates go to the call file only


def test_flux_window_start_is_parsed() -> None:
    ev = parse_flux_message(turn_info("Update", "hi", audio_window_start="3.52"))
    assert isinstance(ev, TurnEvent) and ev.audio_window_start == pytest.approx(3.52)


def test_new_log_lines() -> None:
    assert "bot paused" in (format_record({"type": "barge_pause", "rms": 1500}) or "")
    assert "bot resumed" in (format_record({"type": "barge_resume", "paused_ms": 2500}) or "")
    assert "call continues" in (format_record({"type": "hangup_cancelled"}) or "")
    assert "outcome=not_fit" in (format_record({"type": "ending", "outcome": "not_fit", "silence_s": 3.0}) or "")
    assert (format_record({"type": "ending_blocked"}) or "").startswith("[ending_blocked]")
    plain = format_record({"type": "barge_in", "dropped_frames": 50, "local_pause_ms": None}) or ""
    assert "already paused" not in plain and "1.0s of speech" in plain


# ====================================================================== keyterms
def test_keyterms_are_sent_one_parameter_each_and_url_encoded() -> None:
    url = FluxParams(keyterms=("apartment", "electricity bill", "Northwind Solar")).url()
    assert url.count("keyterm=") == 3
    assert "keyterm=apartment" in url
    assert "keyterm=electricity+bill" in url and "keyterm=Northwind+Solar" in url
    assert "keyterm" not in FluxParams().url()


# ====================================================================== debug recording
def test_recorder_writes_a_playable_private_wav(tmp_path: Path) -> None:
    recorder = InboundRecorder(tmp_path / "rec", "call-1")
    for _ in range(50):
        recorder.write(tone(1000))
    recorder.close()
    recorder.write(FRAME)  # after close: ignored
    recorder.close()
    with wave.open(str(recorder.path), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 8000)
        assert wav.getnframes() == 50 * 160  # exactly one second
    assert recorder.path.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "rec").stat().st_mode & 0o777 == 0o700


def test_recorder_never_raises_when_the_directory_is_unusable(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    recorder = InboundRecorder(blocker / "rec", "call-1")
    recorder.write(FRAME)
    recorder.close()


def test_recordings_older_than_seven_days_are_deleted(tmp_path: Path) -> None:
    now = time.time()

    def make(name: str, age_days: float) -> Path:
        path = tmp_path / name
        path.write_bytes(b"RIFF")
        stamp = now - age_days * 86400
        os.utime(path, (stamp, stamp))
        return path

    old = make("old.wav", 7.5)
    ancient = make("ancient.wav", 40)
    recent = make("recent.wav", 6.5)
    today = make("today.wav", 0)
    other = make("notes.txt", 100)  # only recordings are touched
    assert cleanup_old_recordings(tmp_path) == 2
    assert not old.exists() and not ancient.exists()
    assert recent.exists() and today.exists() and other.exists()
    assert cleanup_old_recordings(tmp_path) == 0
    assert cleanup_old_recordings(tmp_path / "does-not-exist") == 0


def test_recording_is_off_by_default_and_other_new_defaults(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "none.env", environ={})
    assert settings.debug_record_audio is False
    assert settings.barge_vad_enabled is True and settings.barge_vad_mode == 3
    assert settings.barge_min_rms == 500 and settings.barge_confirm_ms == 2500
    assert settings.end_silence_ms == 1500 and settings.end_silence_max_ms == 3000
    on = load_settings(tmp_path / "none.env", environ={"DEBUG_RECORD_AUDIO": "true", "BARGE_VAD_ENABLED": "false"})
    assert on.debug_record_audio is True and on.barge_vad_enabled is False


def test_recordings_directory_is_git_ignored() -> None:
    ignore = (Path(__file__).resolve().parent.parent / ".gitignore").read_text().splitlines()
    assert "data/" in ignore  # recordings live in data/recordings/


# ====================================================================== fixes after the second real call
import httpx  # noqa: E402

from app.intent import caller_reply_kind  # noqa: E402
from app.llm_groq import LLMUsage, TokenBudget  # noqa: E402
from tests.test_agent import delta, groq_with, sse  # noqa: E402


# ---------------------------------------------------------------------- decline / goodbye detection
@pytest.mark.parametrize(
    "text",
    ["No.", "No, thank you.", "Nope, that's all.", "No thanks", "That's all.", "That is it, thanks.",
     "Nothing else.", "I'm good.", "No, I'm good, thank you.", "Not really.", "All set.", "Okay, thank you.",
     "Thanks.", "Thank you very much.", "No no."],
)
def test_declines_are_recognised(text: str) -> None:
    assert caller_reply_kind(text) == "decline", text


@pytest.mark.parametrize(
    "text",
    ["Bye.", "Okay, goodbye!", "Thanks, bye bye.", "Have a nice day.", "No, that's all, bye.",
     "I have to go now.", "Talk to you later.", "Take care."],
)
def test_goodbyes_are_recognised(text: str) -> None:
    assert caller_reply_kind(text) == "goodbye", text


@pytest.mark.parametrize(
    "text",
    [
        "only for own houses, not apartment or rental houses.",  # the real call: a statement, not a decline
        "Yes. What you are providing, what services?",
        "How is the weather today?",
        "I live in apartment.",
        "No, I own the house.",  # "no" as an answer, followed by information
        "There is no roof on the garage.",
        "Not really, how much does it cost?",  # still asking something
        "That's all I know about my roof, what else do you need?",
        "Okay.",
        "Yes.",
        "Wait.",
        "I don't know.",
        "",
    ],
)
def test_statements_questions_and_answers_are_not_declines(text: str) -> None:
    assert caller_reply_kind(text) is None, text


# ---------------------------------------------------------------------- ending guard








# ---------------------------------------------------------------------- no repeated closing question






# ---------------------------------------------------------------------- token budget
def test_token_budget_follows_groq_headers_and_refills_over_a_minute() -> None:
    budget = TokenBudget()
    assert budget.fraction_left(now=0.0) is None and budget.allows_eager(0.4, now=0.0)  # unknown -> allowed
    budget.update({"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "2000"}, now=100.0)
    assert budget.fraction_left(now=100.0) == pytest.approx(0.25)
    assert not budget.allows_eager(0.4, now=100.0)  # 25 % left: no more speculative requests
    assert budget.fraction_left(now=115.0) == pytest.approx(0.5)  # +15 s -> +2000 tokens
    assert budget.allows_eager(0.4, now=115.0)
    assert budget.fraction_left(now=1000.0) == pytest.approx(1.0)  # never above the limit
    budget.exhausted(now=200.0)
    assert budget.fraction_left(now=200.0) == 0.0 and not budget.allows_eager(0.4, now=200.0)
    assert budget.allows_eager(0.0, now=200.0)  # 0 = never skip


def test_token_budget_ignores_missing_or_broken_headers() -> None:
    budget = TokenBudget()
    budget.update({}, now=1.0)
    budget.update({"x-ratelimit-limit-tokens": "abc", "x-ratelimit-remaining-tokens": "5"}, now=1.0)
    budget.update({"x-ratelimit-limit-tokens": "0", "x-ratelimit-remaining-tokens": "0"}, now=1.0)
    assert budget.fraction_left(now=1.0) is None


def test_groq_client_updates_budget_and_counts_cached_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        usage = {"prompt_tokens": 700, "completion_tokens": 30, "prompt_tokens_details": {"cached_tokens": 512}}
        return httpx.Response(
            200,
            content=sse(delta("Hi."), {"choices": [], "usage": usage}),
            headers={"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "1600"},
        )

    async def scenario() -> tuple[LLMUsage, Any]:
        llm = groq_with(handler)
        usage = LLMUsage()
        async for _ in llm.stream([{"role": "user", "content": "hi"}], usage):
            pass
        return usage, llm

    usage, llm = run(scenario)
    assert (usage.prompt_tokens, usage.cached_tokens) == (700, 512)
    assert llm.budget.limit == 8000 and llm.budget.remaining == 1600
    assert not llm.budget.allows_eager(0.4)


def test_groq_429_marks_the_budget_as_exhausted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "Rate limit reached"}},
            headers={"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "900"},
        )

    async def scenario() -> Any:
        llm = groq_with(handler)
        with pytest.raises(Exception):
            async for _ in llm.stream([{"role": "user", "content": "hi"}]):
                pass
        return llm

    assert run(scenario).budget.remaining == 0.0


def test_eager_start_is_skipped_while_the_budget_is_low_and_returns_after() -> None:
    async def scenario() -> Rig:
        allowed = {"value": False}
        rig = Rig(FakeLLM(["First.", "Second."]))
        rig.agent._eager_allowed = lambda: allowed["value"]
        rig.pacer.start()
        rig.event("EagerEndOfTurn", "Hello there.")
        rig.event("EagerEndOfTurn", "Hello there.")
        await asyncio.sleep(0.02)
        assert rig.llm.requests == []  # no speculative request while the budget is low
        rig.event("EndOfTurn", "Hello there.")
        await settle(rig)
        allowed["value"] = True  # budget recovered
        rig.event("StartOfTurn", "x")
        rig.event("EagerEndOfTurn", "And another thing.")
        await until(lambda: len(rig.llm.requests) == 2)
        rig.event("EndOfTurn", "And another thing.")
        await settle(rig)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert len(rig.llm.requests) == 2  # exactly one request per turn
    assert [t["eager"] for t in rig.of("turn")] == [False, True]
    assert rig.agent.eager_skipped == 2 and len(rig.of("eager_skipped")) == 1  # logged once
    assert rig.agent.summary()["eager_skipped"] == 2
    assert (format_record(rig.of("eager_skipped")[0]) or "").startswith("[eager_skipped]")


