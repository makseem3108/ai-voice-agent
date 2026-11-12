"""Nova-3 adapter: Deepgram's transcript pieces become the turn events the agent expects."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlsplit

import pytest

from app.call_log import format_record
from app.config import load_settings
from app.profiles import ProfileError, SttConfig, parse_profile
from app.providers import check_providers, describe_stt, make_stt, stt_eager
from app.speech import END_OF_TURN, START_OF_TURN, UPDATE, STTProvider, SttError, TurnEvent
from app.stt_deepgram import AudioClock, DeepgramParams, Transcript, UtteranceEnd
from app.stt_flux import FluxStream
from app.stt_nova import NovaStream, NovaTurnTracker
from tests.helpers import PROFILE, FakeLLM, Rig, reply, run, until
from tests.test_stt_deepgram import FakeWebSocket


def piece(text: str, *, final: bool = False, speech_final: bool = False, start: float = 0.0, duration: float = 1.0, last: Optional[float] = None) -> Transcript:
    return Transcript(text=text, is_final=final, speech_final=speech_final, start=start, duration=duration, confidence=0.9, last_word_end=last)


def kinds(events: list[TurnEvent]) -> list[str]:
    return [e.event for e in events]


# ====================================================================== turn tracker
def test_one_short_turn() -> None:
    t = NovaTurnTracker()
    first = t.feed(piece("שלום"))
    assert kinds(first) == [START_OF_TURN, UPDATE] and first[0].transcript == "שלום"
    assert kinds(t.feed(piece("שלום אני רוצה"))) == [UPDATE]
    end = t.feed(piece("שלום, אני רוצה תור.", final=True, speech_final=True, start=0.2, duration=2.0, last=2.1))
    assert kinds(end) == [UPDATE, END_OF_TURN]
    eot = end[-1]
    assert eot.transcript == "שלום, אני רוצה תור." and eot.turn_index == 0 and eot.eot_confidence == 1.0
    assert eot.last_word_end == 2.1 and eot.speech_end == 2.1 and eot.audio_window_end == 2.2
    assert eot.details == {
        "ended_by": "speech_final",
        "segments": [{"interim": "שלום אני רוצה", "final": "שלום, אני רוצה תור."}],
        "lost_words": [],
    }
    assert t.open is False


def test_final_pieces_are_joined_until_speech_final() -> None:
    """Seen live: one Arabic sentence arrived as two final pieces; only the second had speech_final."""
    t = NovaTurnTracker()
    t.feed(piece("مرحبا بدي"))
    assert kinds(t.feed(piece("مرحبا بدي احجز موعد عند", final=True, start=0.0, duration=3.0, last=2.9))) == [UPDATE]
    assert t.open  # not over: no speech_final yet
    update = t.feed(piece("طبيب الاسنان"))[-1]
    assert update.transcript == "مرحبا بدي احجز موعد عند طبيب الاسنان"  # text so far = finals + current guess
    end = t.feed(piece("طبيب الاسنان يوم الثلاثاء الصبح", final=True, speech_final=True, start=3.0, duration=2.5, last=5.4))[-1]
    assert end.event == END_OF_TURN
    assert end.transcript == "مرحبا بدي احجز موعد عند طبيب الاسنان يوم الثلاثاء الصبح"
    assert len(end.details["segments"]) == 2 and end.last_word_end == 5.4 and end.audio_window_start == 0.0


def test_words_dropped_between_interim_and_final_are_reported() -> None:
    """The dropped word from the phase 2c review: the guess had it, the final text did not."""
    t = NovaTurnTracker()
    t.feed(piece("عند طبيب الاسنان يوم"))
    end = t.feed(piece("عند الاسنان يوم الثلاثاء", final=True, speech_final=True))[-1]
    assert end.details["lost_words"] == ["طبيب"]
    assert end.details["segments"] == [{"interim": "عند طبيب الاسنان يوم", "final": "عند الاسنان يوم الثلاثاء"}]
    line = format_record({"type": "stt_event", "event": "EndOfTurn", "details": end.details})
    assert line == "[stt_turn] ended_by=speech_final pieces=1 words in the interim guess but not in the final text: طبيب"
    clean = format_record({"type": "stt_event", "event": "EndOfTurn", "details": {"ended_by": "utterance_end", "segments": [{}, {}], "lost_words": []}})
    assert clean == "[stt_turn] ended_by=utterance_end pieces=2 no words lost between interim and final"


def test_half_heard_words_and_joined_particles_are_not_reported_as_lost() -> None:
    """Seen live: the guess "اساعد" became "اساعدك", and "והביקור" became "הביקור"."""
    t = NovaTurnTracker()
    t.feed(piece("شو بقدر اساعد"))
    assert t.feed(piece("شو بقدر اساعدك", final=True, speech_final=True))[-1].details["lost_words"] == []
    t.feed(piece("והביקור עולה 250"))
    assert t.feed(piece("הביקור עולה 250 שקלים.", final=True, speech_final=True))[-1].details["lost_words"] == []


def test_agent_stream_asks_for_words_not_formatted_numbers() -> None:
    """Live: with smart_format Nova-3 wrote "יום שלישי" (Tuesday) as "יום 3º"."""
    agent_params = DeepgramParams(language="he", smart_format=False)
    query = dict(query_of(agent_params))
    assert "smart_format" not in query and query["punctuate"] == "true"
    assert dict(query_of(DeepgramParams()))["smart_format"] == "true"  # phase 2a transcription mode keeps it


def test_a_word_that_only_moved_to_another_piece_is_not_lost() -> None:
    t = NovaTurnTracker()
    t.feed(piece("one two three"))
    t.feed(piece("One two.", final=True))
    t.feed(piece("three four"))
    end = t.feed(piece("Three four.", final=True, speech_final=True))[-1]
    assert end.details["lost_words"] == []  # compared with the whole turn, ignoring case and punctuation


def test_utterance_end_is_the_backup_end_of_turn() -> None:
    t = NovaTurnTracker()
    t.feed(piece("כן", final=True, start=0.2, duration=0.5, last=0.6))  # final, but endpointing never fired (noisy line)
    end = t.feed(UtteranceEnd(last_word_end=0.6))
    assert kinds(end) == [END_OF_TURN] and end[0].transcript == "כן"
    assert end[0].details["ended_by"] == "utterance_end" and end[0].eot_confidence == 0.5
    assert t.feed(UtteranceEnd(last_word_end=0.6)) == []  # nothing open: ignored


def test_late_utterance_end_for_earlier_speech_does_not_cut_the_next_turn() -> None:
    """Seen live: the backup signal for the previous sentence arrived just after the next one had begun,
    which produced a bogus one-word turn ("موعد") followed by the real sentence."""
    t = NovaTurnTracker()
    t.feed(piece("شو بقدر اساعدك", final=True, speech_final=True, start=4.0, duration=2.2, last=6.17))
    assert kinds(t.feed(piece("موعد", start=7.4, duration=0.5))) == [START_OF_TURN, UPDATE]  # the next sentence begins
    assert t.feed(UtteranceEnd(last_word_end=6.17)) == [] and t.open  # about the old sentence: ignored
    end = t.feed(piece("موعدك يوم الثلاثا", final=True, speech_final=True, start=7.4, duration=3.0, last=10.2))[-1]
    assert end.event == END_OF_TURN and end.transcript == "موعدك يوم الثلاثا"
    # one that is about the open turn still ends it
    t.feed(piece("نعم", final=True, start=12.0, duration=0.5, last=12.4))
    assert kinds(t.feed(UtteranceEnd(last_word_end=12.4))) == [END_OF_TURN]


def test_timeout_is_our_own_last_resort_and_keeps_an_unfinalized_guess() -> None:
    t = NovaTurnTracker()
    t.feed(piece("אני רוצה לקבוע"))  # only ever a guess
    end = t.timeout()
    assert kinds(end) == [END_OF_TURN] and end[0].transcript == "אני רוצה לקבוע"
    assert end[0].details == {"ended_by": "timeout", "segments": [{"interim": "אני רוצה לקבוע", "final": None}], "lost_words": []}
    assert t.timeout() == []


def test_empty_results_never_open_or_end_a_turn() -> None:
    t = NovaTurnTracker()
    assert t.feed(piece("")) == [] and t.feed(piece("", final=True, speech_final=True)) == []
    assert t.feed(UtteranceEnd(last_word_end=-1)) == [] and t.open is False
    # the trailing empty speech_final Deepgram sends after a turn does nothing either
    t.feed(piece("שלום", final=True, speech_final=True))
    assert t.feed(piece("", final=True, speech_final=True)) == []


def test_turns_are_numbered_and_independent() -> None:
    t = NovaTurnTracker()
    first = t.feed(piece("אחת", final=True, speech_final=True))[-1]
    second = t.feed(piece("שתיים", final=True, speech_final=True))
    assert (first.turn_index, second[0].turn_index, second[-1].turn_index) == (0, 1, 1)
    assert second[0].event == START_OF_TURN and second[-1].transcript == "שתיים"


# ====================================================================== request parameters
def query_of(params: DeepgramParams) -> list[tuple[str, str]]:
    return parse_qsl(urlsplit(params.url()).query)


def test_request_carries_language_endpointing_and_keyterms_in_any_script() -> None:
    params = DeepgramParams(language="he", endpointing_ms=300, utterance_end_ms=1000, keyterms=("קופת חולים", "תור"))
    query = query_of(params)
    assert ("model", "nova-3") in query and ("language", "he") in query
    assert ("endpointing", "300") in query and ("utterance_end_ms", "1000") in query and ("interim_results", "true") in query
    assert ("encoding", "linear16") in query and ("sample_rate", "8000") in query
    assert [v for k, v in query if k == "keyterm"] == ["קופת חולים", "תור"]
    assert "language_hint" not in dict(query)
    assert params.url().startswith("wss://api.deepgram.com/v1/listen?") and "Token" not in params.url()


def test_multi_mode_sends_language_hints() -> None:
    params = DeepgramParams(language="multi", language_hints=("ar", "he"))
    query = query_of(params)
    assert ("language", "multi") in query and [v for k, v in query if k == "language_hint"] == ["ar", "he"]
    assert "language_hint" not in dict(query_of(DeepgramParams(language="ar", language_hints=("ar", "he"))))  # hints only with multi


def test_phase_2a_request_is_unchanged() -> None:
    assert "keyterm" not in DeepgramParams().url() and dict(query_of(DeepgramParams()))["endpointing"] == "400"


# ====================================================================== the stream
def results(text: str, *, final: bool = False, speech_final: bool = False, start: float = 0.0, duration: float = 1.0) -> str:
    words = [{"word": w, "start": start, "end": start + duration} for w in text.split()]
    return json.dumps({
        "type": "Results", "is_final": final, "speech_final": speech_final, "start": start, "duration": duration,
        "channel": {"alternatives": [{"transcript": text, "confidence": 0.9, "words": words}]},
    })


def test_stream_is_an_stt_provider_and_translates_messages() -> None:
    async def scenario() -> tuple[list[Any], NovaStream, FakeWebSocket, list[Any]]:
        ws = FakeWebSocket()
        seen: list[Any] = []
        connects: list[Any] = []

        async def connect(url: str, headers: dict[str, str]) -> FakeWebSocket:
            connects.append((url, headers))
            return ws

        stream = NovaStream("dg-test-key", DeepgramParams(language="he"), lambda ev, wall: seen.append(ev), connect=connect)
        assert isinstance(stream, STTProvider)
        assert (stream.name, stream.model, stream.supports_eager, stream.failed) == ("nova3", "nova-3/he", False, False)
        stream.start()
        stream.send_audio(b"\x00" * 320)
        for message in (
            json.dumps({"type": "SpeechStarted", "timestamp": 0.1}),
            results("שלום"),
            results("שלום, יש לי שאלה.", final=True, speech_final=True, duration=1.5),
            json.dumps({"type": "UtteranceEnd", "last_word_end": 1.5}),
        ):
            ws._incoming.put_nowait(message)
        await asyncio.sleep(0.05)
        await stream.close()
        return seen, stream, ws, connects

    seen, stream, ws, connects = run(scenario)
    assert kinds(seen) == [START_OF_TURN, UPDATE, UPDATE, END_OF_TURN]  # the late UtteranceEnd adds nothing
    assert seen[-1].transcript == "שלום, יש לי שאלה." and seen[-1].last_word_end == 1.5
    assert stream.turns_by == {"speech_final": 1, "utterance_end": 0, "timeout": 0}
    ((url, headers),) = connects
    assert headers == {"Authorization": "Token dg-test-key"} and "dg-test-key" not in url and "language=he" in url
    assert b"\x00" * 320 in ws.sent and stream.clock.total_seconds == pytest.approx(0.02)


def test_stream_ends_the_turn_itself_when_deepgram_stays_silent() -> None:
    async def scenario() -> tuple[list[Any], NovaStream]:
        ws = FakeWebSocket()
        seen: list[Any] = []

        async def connect(url: str, headers: dict[str, str]) -> FakeWebSocket:
            return ws

        stream = NovaStream("k", DeepgramParams(language="ar"), lambda ev, wall: seen.append(ev), connect=connect, turn_timeout_s=0.15)
        stream.start()
        ws._incoming.put_nowait(results("بدي موعد", final=True))  # final, but no speech_final and no UtteranceEnd ever
        await asyncio.sleep(0.05)
        assert END_OF_TURN not in kinds(seen)
        await asyncio.sleep(0.2)
        await stream.close()
        return seen, stream

    seen, stream = run(scenario)
    assert kinds(seen)[-1] == END_OF_TURN and seen[-1].transcript == "بدي موعد" and seen[-1].details["ended_by"] == "timeout"
    assert stream.turns_by["timeout"] == 1


def test_words_keep_pushing_our_timeout_out() -> None:
    async def scenario() -> list[Any]:
        ws = FakeWebSocket()
        seen: list[Any] = []

        async def connect(url: str, headers: dict[str, str]) -> FakeWebSocket:
            return ws

        stream = NovaStream("k", DeepgramParams(language="he"), lambda ev, wall: seen.append(ev), connect=connect, turn_timeout_s=0.15)
        stream.start()
        for text in ("אני", "אני רוצה", "אני רוצה לקבוע", "אני רוצה לקבוע תור"):
            ws._incoming.put_nowait(results(text))
            await asyncio.sleep(0.08)  # slower than half the timeout, never longer than it
        assert END_OF_TURN not in kinds(seen)  # still talking
        ws._incoming.put_nowait(results("אני רוצה לקבוע תור.", final=True, speech_final=True))
        await asyncio.sleep(0.3)  # the timer must be cancelled: no second EndOfTurn
        await stream.close()
        return seen

    seen = run(scenario)
    assert kinds(seen).count(END_OF_TURN) == 1 and seen[-1].details["ended_by"] == "speech_final"


def test_connection_failure_is_reported_once_as_an_error() -> None:
    async def scenario() -> tuple[list[Any], NovaStream]:
        seen: list[Any] = []

        async def connect(url: str, headers: dict[str, str]) -> FakeWebSocket:
            raise OSError("network unreachable")

        stream = NovaStream("k", DeepgramParams(language="he"), lambda ev, wall: seen.append(ev), connect=connect)
        stream.start()
        await asyncio.sleep(0.05)
        stream.send_audio(b"\x00" * 320)  # dropped silently
        await stream.close()
        return seen, stream

    seen, stream = run(scenario)
    assert len(seen) == 1 and isinstance(seen[0], SttError) and stream.failed is True


# ====================================================================== profile + factory
def profile_data(**extra: Any) -> dict[str, Any]:
    return {
        "business_name": "Acme", "agent_role": "the receptionist", "goal": "Take a message.", "greeting": "Hello.",
        "slots": [{"name": "ok", "type": "yes_no", "question": "Okay?"}], "knowledge": "Nothing.", "tone": "friendly",
        "closing_qualified": "Bye!", "closing_not_fit": "Sorry, bye!", **extra,
    }


def test_profile_selects_nova_with_language_or_multi_and_hints(tmp_path: Path) -> None:
    async def scenario() -> None:
        s = load_settings(tmp_path / "none.env", environ={"DEEPGRAM_API_KEY": "dg", "NOVA_ENDPOINTING_MS": "250"})
        hebrew = dataclasses.replace(PROFILE, language="he", stt=SttConfig(provider="nova3"), keyterms=("תור",))
        stream = make_stt(hebrew, s, lambda ev, wall: None, AudioClock())
        assert isinstance(stream, NovaStream) and stream.model == "nova-3/he"
        assert stream._params == DeepgramParams(language="he", endpointing_ms=250, utterance_end_ms=1000, keyterms=("תור",), smart_format=False)
        assert describe_stt(hebrew, s) == "nova-3/he" and stt_eager(hebrew, s) is False

        mixed = parse_profile("m", profile_data(language="ar", stt={"provider": "nova3", "language": "multi", "language_hints": ["ar", "he"]}))
        assert mixed.stt == SttConfig("nova3", "", "multi", ("ar", "he"))
        multi = make_stt(mixed, s, lambda ev, wall: None, AudioClock())
        assert multi._params.language == "multi" and multi._params.language_hints == ("ar", "he")
        assert describe_stt(mixed, s) == "nova-3/multi[ar,he]"

        dialect = dataclasses.replace(hebrew, language="ar-JO")
        assert make_stt(dialect, s, lambda ev, wall: None, AudioClock())._params.language == "ar-JO"
        check_providers(dataclasses.replace(hebrew, tts=PROFILE.tts.__class__(provider="azure")), dataclasses.replace(s, azure_speech_key="k", azure_speech_region="r"))

        english = make_stt(PROFILE, s, lambda ev, wall: None, AudioClock())
        assert isinstance(english, FluxStream) and describe_stt(PROFILE, s) == "flux-general-en" and stt_eager(PROFILE, s) is True

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "stt, message",
    [
        ({"provider": "nova3", "language": "ar", "language_hints": ["ar", "he"]}, "only works with stt.language: multi"),
        ({"provider": "nova3", "language": "multi", "language_hints": "ar"}, "must be a list of language codes"),
        ({"provider": "nova3", "language": "multi", "language_hints": ["Arabic"]}, "must be a list of language codes"),
        ({"provider": "nova3", "language": "multi", "language_hints": []}, "must be a list of language codes"),
    ],
)
def test_bad_language_hints_are_rejected(stt: dict[str, Any], message: str) -> None:
    with pytest.raises(ProfileError) as info:
        parse_profile("t", profile_data(language="ar", stt=stt))
    assert message in str(info.value)


def test_default_settings() -> None:
    s = load_settings(Path("/nonexistent.env"), environ={})
    assert (s.nova_endpointing_ms, s.nova_utterance_end_ms, s.nova_turn_timeout_ms) == (500, 1000, 2500)


# ====================================================================== with the agent
def test_agent_runs_on_nova_events_without_eager_and_logs_the_details() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM([reply("באיזה יום נוח לך?")]), eager=False)
        tracker = NovaTurnTracker()
        rig.pacer.start()
        import time

        for message in (piece("שלום"), piece("שלום אני רוצה תור"), piece("שלום, אני רוצה תור.", final=True, speech_final=True, duration=2.0, last=1.9)):
            for event in tracker.feed(message):
                rig.agent.on_stt_event(event, time.time())
        await until(lambda: rig.agent._job is None and len(rig.of("bot")) == 1)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.of("user")[0]["text"] == "שלום, אני רוצה תור." and rig.bot_lines() == ["באיזה יום נוח לך?"]
    assert len(rig.llm.requests) == 1  # no speculative request: one reply, started at EndOfTurn
    assert rig.of("turn")[0]["eager"] is False and rig.of("turn")[0]["stt_ms"] is not None
    ends = [r for r in rig.of("stt_event") if r["event"] == "EndOfTurn"]
    assert ends[0]["details"]["ended_by"] == "speech_final" and ends[0]["last_word_end"] == 1.9
    assert (format_record(ends[0]) or "").startswith("[stt_turn] ended_by=speech_final pieces=1")


# ====================================================================== turn merging (Nova-3 only)
def test_caller_who_goes_on_before_the_reply_starts_is_answered_as_one_turn() -> None:
    """Nova-3 ends a turn on a 0.5 s pause; callers pause that long between sentences."""

    async def scenario() -> Rig:
        llm = FakeLLM([reply("Reply to the first half only."), reply("באיזה יום נוח לך?")])
        llm.hold = asyncio.Event()  # the first request is still running when the caller continues
        rig = Rig(llm, eager=False, merge_turns=True)
        rig.pacer.start()
        rig.event("StartOfTurn", "אני")
        rig.event("EndOfTurn", "אני רוצה לקבוע תור.")
        await until(lambda: len(llm.requests) == 1)
        await asyncio.sleep(0.05)
        rig.event("StartOfTurn", "אם")  # goes on speaking; nothing of the reply was audible yet
        assert rig.agent._job is None and rig.frames == []
        await asyncio.sleep(0.01)
        assert llm.cancelled == 1  # the pending request was dropped
        llm.hold.set()
        rig.event("EndOfTurn", "אם אפשר ביום שלישי.")
        await until(lambda: rig.agent._job is None and len(rig.of("bot")) == 1)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.bot_lines() == ["באיזה יום נוח לך?"]  # one reply, to the whole thing
    assert rig.llm.requests[-1][-2] == {"role": "user", "content": "אני רוצה לקבוע תור. אם אפשר ביום שלישי."}
    (merge,) = rig.of("turn_merge")
    assert merge["parts"] == ["אני רוצה לקבוע תור.", "אם אפשר ביום שלישי."] and merge["text"] == "אני רוצה לקבוע תור. אם אפשר ביום שלישי."
    assert 30 <= merge["gap_ms"] < 500 and merge["turn"] == 2
    assert rig.agent.summary()["turn_merges"] == 1 and rig.agent.barge_ins == 0
    line = format_record(merge)
    assert line is not None and line.startswith("[turn_merge] the caller went on speaking") and "answered as one turn" in line


def test_three_parts_are_merged_too() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM([reply("one"), reply("two"), reply("All three.")])
        llm.hold = asyncio.Event()
        rig = Rig(llm, eager=False, merge_turns=True)
        rig.pacer.start()
        for index, text in enumerate(["First.", "Second.", "Third."]):
            rig.event("StartOfTurn", text)
            if index == 2:
                llm.hold.set()
            rig.event("EndOfTurn", text)
            await until(lambda: len(llm.requests) == index + 1)
        await until(lambda: rig.agent._job is None and len(rig.of("bot")) == 1)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.llm.requests[-1][-2]["content"] == "First. Second. Third." and rig.bot_lines() == ["All three."]
    assert [m["text"] for m in rig.of("turn_merge")] == ["First. Second.", "Second. Third."] and rig.agent.turn_merges == 2


def test_speaking_after_the_reply_started_is_a_barge_in_not_a_merge() -> None:
    async def scenario() -> Rig:
        from tests.helpers import FakeTTS

        rig = Rig(FakeLLM([reply("A long answer that is already playing."), reply("Sure.")]), FakeTTS(frames_per_sentence=60), eager=False, merge_turns=True)
        rig.pacer.start()
        rig.event("StartOfTurn", "Hi")
        rig.event("EndOfTurn", "Hi.")
        await until(lambda: rig.pacer.speaking and len(rig.frames) > 3)
        rig.event("StartOfTurn", "Wait")
        rig.event("EndOfTurn", "Wait.")
        await until(lambda: rig.agent._job is None and len(rig.of("bot")) == 2)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.of("turn_merge") == [] and rig.agent.barge_ins == 1


def test_flux_calls_do_not_log_merges_and_nova_streams_ask_for_them() -> None:
    async def scenario() -> Rig:
        llm = FakeLLM([reply("one"), reply("Both.")])
        llm.hold = asyncio.Event()
        rig = Rig(llm, eager=False)  # merge_turns off: the English path, as before
        rig.pacer.start()
        rig.event("StartOfTurn", "First")
        rig.event("EndOfTurn", "First.")
        await until(lambda: len(llm.requests) == 1)
        rig.event("StartOfTurn", "Second")
        llm.hold.set()
        rig.event("EndOfTurn", "Second.")
        await until(lambda: rig.agent._job is None and len(rig.of("bot")) == 1)
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.of("turn_merge") == [] and rig.agent.turn_merges == 0
    assert rig.llm.requests[-1][-2]["content"] == "First. Second."  # the history was always joined; only the log is new
    assert NovaStream.merges_turns is True and getattr(FluxStream, "merges_turns", False) is False
