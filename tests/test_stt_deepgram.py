"""Tests for Deepgram message parsing, latency maths and the stream wrapper.

No network: the WebSocket is replaced by :class:`FakeWebSocket`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import pytest

from app.call_log import CallRecorder, format_record
from app.config import Settings, load_settings
from app.stt_deepgram import (
    AudioClock,
    DeepgramParams,
    DeepgramStream,
    SpeechStarted,
    StreamMetadata,
    SttError,
    SttEvent,
    Transcript,
    TranscriptTracker,
    Unknown,
    UtteranceEnd,
    parse_message,
)

FRAME = bytes(320)  # 20 ms of silence
FAKE_KEY = "test-key-not-a-real-secret"


def results(
    text: str,
    *,
    is_final: bool,
    speech_final: bool = False,
    start: float = 0.0,
    duration: float = 1.0,
    word_end: Optional[float] = None,
) -> str:
    words = [] if word_end is None else [{"word": "x", "start": word_end - 0.2, "end": word_end}]
    return json.dumps(
        {
            "type": "Results",
            "start": start,
            "duration": duration,
            "is_final": is_final,
            "speech_final": speech_final,
            "channel": {"alternatives": [{"transcript": text, "confidence": 0.97, "words": words}]},
        }
    )


# ------------------------------------------------------------------ message parsing
def test_parse_interim_result() -> None:
    ev = parse_message(results("Hi I'd like", is_final=False, start=0.0, duration=1.2, word_end=1.1))
    assert isinstance(ev, Transcript)
    assert (ev.text, ev.is_final, ev.speech_final) == ("Hi I'd like", False, False)
    assert ev.end == pytest.approx(1.2)
    assert ev.last_word_end == pytest.approx(1.1)
    assert ev.confidence == pytest.approx(0.97)


def test_parse_final_with_speech_final() -> None:
    ev = parse_message(results("Hello.", is_final=True, speech_final=True, start=2.0, duration=1.5, word_end=3.0))
    assert isinstance(ev, Transcript)
    assert ev.is_final and ev.speech_final
    assert ev.end == pytest.approx(3.5)


def test_parse_result_accepts_bytes_and_strips_text() -> None:
    ev = parse_message(results("  hello  ", is_final=True).encode())
    assert isinstance(ev, Transcript) and ev.text == "hello"


def test_parse_result_without_words_or_alternatives() -> None:
    ev = parse_message(json.dumps({"type": "Results", "start": 1.0, "duration": 0.5, "is_final": True}))
    assert isinstance(ev, Transcript)
    assert ev.text == "" and ev.last_word_end is None


def test_parse_vad_and_metadata_events() -> None:
    assert parse_message('{"type":"SpeechStarted","channel":[0,1],"timestamp":3.2}') == SpeechStarted(3.2)
    assert parse_message('{"type":"UtteranceEnd","channel":[0,1],"last_word_end":2.39}') == UtteranceEnd(2.39)
    meta = parse_message('{"type":"Metadata","request_id":"abc","duration":12.5,"channels":1}')
    assert meta == StreamMetadata(request_id="abc", duration=12.5)


def test_parse_error_unknown_and_garbage_never_raise() -> None:
    err = parse_message('{"type":"Error","description":"bad things"}')
    assert isinstance(err, SttError) and "bad things" in err.message
    assert parse_message('{"type":"SomethingNew"}') == Unknown("SomethingNew")
    assert isinstance(parse_message("not json"), SttError)
    assert isinstance(parse_message(b"\xff\xfe"), SttError)
    assert isinstance(parse_message("[1,2,3]"), SttError)


# ------------------------------------------------------------------ request parameters
def test_url_has_requested_params_and_no_credentials() -> None:
    url = DeepgramParams().url()
    assert url.startswith("wss://api.deepgram.com/v1/listen?")
    for part in (
        "model=nova-3", "language=en", "encoding=linear16", "sample_rate=8000", "channels=1",
        "interim_results=true", "endpointing=400", "utterance_end_ms=1000", "vad_events=true",
        "smart_format=true",
    ):
        assert part in url
    assert "token" not in url.lower() and "key" not in url.lower()


# ------------------------------------------------------------------ audio clock
def make_clock(n_frames: int, t0: float = 1000.0, frame_s: float = 0.02) -> AudioClock:
    """n frames of 20 ms, frame i arriving at t0 + i*20ms."""
    clock = AudioClock()
    for i in range(n_frames):
        clock.mark(frame_s, t0 + i * frame_s)
    return clock


def test_clock_empty() -> None:
    clock = AudioClock()
    assert clock.total_seconds == 0.0
    assert clock.wall_time_at(1.0) is None
    assert clock.latency_ms(1.0, 5.0) is None


def test_clock_total_seconds_is_billable_audio() -> None:
    assert make_clock(250).total_seconds == pytest.approx(5.0)


def test_clock_maps_audio_time_to_the_frame_that_contained_it() -> None:
    clock = make_clock(100)
    assert clock.wall_time_at(0.0) == pytest.approx(1000.0)  # first frame
    assert clock.wall_time_at(0.02) == pytest.approx(1000.0)  # boundary belongs to frame 0
    assert clock.wall_time_at(0.021) == pytest.approx(1000.02)  # frame 1
    assert clock.wall_time_at(1.0) == pytest.approx(1000.98)  # frame 49 covers 0.98-1.00
    assert clock.wall_time_at(99.0) == pytest.approx(1001.98)  # beyond the end -> last frame


def test_clock_uses_real_arrival_times_not_an_ideal_timeline() -> None:
    clock = AudioClock()
    clock.mark(0.02, 10.000)
    clock.mark(0.02, 10.500)  # this frame was stuck in the network for half a second
    clock.mark(0.02, 10.520)
    assert clock.wall_time_at(0.03) == pytest.approx(10.500)
    assert clock.latency_ms(0.03, 11.000) == pytest.approx(500.0)


def test_latency_is_arrival_minus_time_the_audio_was_heard() -> None:
    clock = make_clock(200)  # 4 s of audio, starting at wall 1000.0
    # last word ended at audio 2.00 s -> frame 99, heard at wall 1001.98
    assert clock.latency_ms(2.0, 1002.50) == pytest.approx(520.0)
    assert clock.latency_ms(2.0, 1001.00) == 0.0  # never negative


# ------------------------------------------------------------------ transcript tracker
def test_tracker_interim_is_logged_but_not_counted() -> None:
    tracker = TranscriptTracker(make_clock(100))
    records = tracker.on_event(parse_message(results("Hi I'd", is_final=False, word_end=0.8)), 1001.0)
    assert [r["type"] for r in records] == ["interim"]
    assert tracker.finals == 0 and tracker.latencies_ms == []


def test_tracker_final_latency_uses_last_word_end() -> None:
    tracker = TranscriptTracker(make_clock(200))
    ev = parse_message(results("Book it.", is_final=True, start=0.0, duration=2.5, word_end=2.0))
    (final,) = tracker.on_event(ev, 1002.50)
    assert final["type"] == "final" and final["text"] == "Book it."
    assert final["last_word_end"] == pytest.approx(2.0)
    assert final["latency_ms"] == pytest.approx(520.0)
    assert tracker.finals == 1


def test_tracker_final_without_word_timings_falls_back_to_segment_end() -> None:
    tracker = TranscriptTracker(make_clock(200))
    (final,) = tracker.on_event(parse_message(results("Yes.", is_final=True, start=1.0, duration=1.0)), 1002.30)
    # segment end = 2.0 s -> heard at 1001.98
    assert final["latency_ms"] == pytest.approx(320.0)


def test_tracker_speech_end_reports_silence_after_last_word() -> None:
    tracker = TranscriptTracker(make_clock(200))
    ev = parse_message(
        results("Tuesday.", is_final=True, speech_final=True, start=0.0, duration=2.42, word_end=2.0)
    )
    final, speech_end = tracker.on_event(ev, 1002.60)
    assert final["type"] == "final" and final["speech_final"] is True
    assert speech_end == {"type": "speech_end", "silence_s": 0.42}
    assert format_record(speech_end) == "[speech_end] after 0.42s silence"


def test_tracker_speech_final_on_empty_result_closes_the_open_utterance() -> None:
    tracker = TranscriptTracker(make_clock(300))
    tracker.on_event(parse_message(results("Hello", is_final=True, start=0.0, duration=1.5, word_end=1.2)), 1001.7)
    records = tracker.on_event(
        parse_message(results("", is_final=True, speech_final=True, start=1.5, duration=0.3)), 1002.0
    )
    assert records == [{"type": "speech_end", "silence_s": 0.6}]
    assert tracker.finals == 1  # the empty final is not counted


def test_tracker_ignores_speech_final_when_nobody_spoke() -> None:
    tracker = TranscriptTracker(make_clock(100))
    assert tracker.on_event(parse_message(results("", is_final=True, speech_final=True)), 1001.0) == []
    assert tracker.finals == 0


def test_tracker_only_one_speech_end_per_utterance() -> None:
    tracker = TranscriptTracker(make_clock(300))
    first = results("One.", is_final=True, speech_final=True, start=0.0, duration=1.4, word_end=1.0)
    assert len(tracker.on_event(parse_message(first), 1001.5)) == 2
    again = results("", is_final=True, speech_final=True, start=1.4, duration=1.0)
    assert tracker.on_event(parse_message(again), 1002.5) == []


def test_tracker_avg_and_max_latency() -> None:
    tracker = TranscriptTracker(make_clock(500))
    tracker.on_event(parse_message(results("a", is_final=True, word_end=1.0)), 1000.98 + 0.300)
    tracker.on_event(parse_message(results("b", is_final=True, word_end=2.0)), 1001.98 + 0.700)
    assert tracker.finals == 2
    assert tracker.avg_latency_ms == pytest.approx(500.0)
    assert tracker.max_latency_ms == pytest.approx(700.0)


def test_tracker_no_finals_means_no_latency() -> None:
    tracker = TranscriptTracker(AudioClock())
    assert tracker.avg_latency_ms is None and tracker.max_latency_ms is None


def test_tracker_other_events() -> None:
    tracker = TranscriptTracker(make_clock(100))
    assert tracker.on_event(SpeechStarted(0.5), 1.0) == [{"type": "speech_start", "audio_t": 0.5}]
    assert tracker.on_event(UtteranceEnd(1.25), 1.0) == [{"type": "utterance_end", "last_word_end": 1.25}]
    assert tracker.on_event(StreamMetadata("req-1", 9.5), 1.0) == []
    assert (tracker.request_id, tracker.deepgram_duration) == ("req-1", 9.5)
    assert tracker.on_event(SttError("boom"), 1.0) == [{"type": "stt_error", "message": "boom"}]
    assert tracker.on_event(Unknown("x"), 1.0) == []


# ------------------------------------------------------------------ log formatting
def test_format_records() -> None:
    assert format_record({"type": "interim", "text": "Hi I'd like"}) == "[interim] Hi I'd like"
    final = {"type": "final", "text": "Hi.", "latency_ms": 512.4}
    assert format_record(final) == "[final]   Hi.  (stt latency 512 ms)"
    assert format_record({"type": "final", "text": "Hi.", "latency_ms": None}) == "[final]   Hi."
    assert format_record({"type": "call_start"}) is None


# ------------------------------------------------------------------ stream (fake websocket)
class FakeWebSocket:
    """Stands in for a websockets connection. Records what was sent."""

    def __init__(self, incoming: Optional[list[str]] = None, close_on_closestream: bool = True) -> None:
        self.sent: list[Union[str, bytes]] = []
        self.closed = False
        self._incoming: asyncio.Queue[Union[str, BaseException]] = asyncio.Queue()
        for message in incoming or []:
            self._incoming.put_nowait(message)
        self._close_on_closestream = close_on_closestream
        self.fail_next_send: Optional[BaseException] = None

    async def send(self, message: Union[str, bytes]) -> None:
        if self.fail_next_send is not None:
            raise self.fail_next_send
        self.sent.append(message)
        if isinstance(message, str) and json.loads(message).get("type") == "CloseStream":
            self._incoming.put_nowait('{"type":"Metadata","request_id":"req-9","duration":0.06}')
            if self._close_on_closestream:
                self._incoming.put_nowait(ConnectionError("closed OK"))

    async def recv(self) -> Union[str, bytes]:
        item = await self._incoming.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def drop(self) -> None:
        """Simulate Deepgram disconnecting unexpectedly."""
        self._incoming.put_nowait(ConnectionError("connection lost"))

    async def close(self) -> None:
        self.closed = True

    @property
    def control(self) -> list[str]:
        return [json.loads(m)["type"] for m in self.sent if isinstance(m, str)]

    @property
    def audio(self) -> list[bytes]:
        return [m for m in self.sent if isinstance(m, bytes)]


class Harness:
    def __init__(self, ws: Optional[FakeWebSocket] = None, connect_error: Optional[BaseException] = None, **kw: Any) -> None:
        self.ws = ws or FakeWebSocket()
        self.connect_error = connect_error
        self.connect_calls: list[tuple[str, dict[str, str]]] = []
        self.events: list[SttEvent] = []
        self.stream = DeepgramStream(
            FAKE_KEY, DeepgramParams(), lambda ev, _t: self.events.append(ev), connect=self._connect, **kw
        )

    async def _connect(self, url: str, headers: dict[str, str]) -> FakeWebSocket:
        self.connect_calls.append((url, headers))
        if self.connect_error is not None:
            raise self.connect_error
        return self.ws


def test_stream_sends_frames_in_order_then_closestream() -> None:
    async def scenario() -> Harness:
        h = Harness()
        h.stream.start()
        frames = [bytes([i]) * 320 for i in range(3)]
        for i, frame in enumerate(frames):
            h.stream.send_audio(frame, 100.0 + i * 0.02)
        await h.stream.close()
        assert h.ws.audio == frames
        return h

    h = asyncio.run(scenario())
    assert h.ws.sent[-1] == '{"type": "CloseStream"}'
    assert h.ws.control == ["CloseStream"]
    assert h.ws.closed
    assert not h.stream.failed
    assert h.stream.clock.total_seconds == pytest.approx(0.06)
    assert StreamMetadata("req-9", 0.06) in h.events  # results after CloseStream are still read


def test_stream_authenticates_with_header_not_url() -> None:
    async def scenario() -> Harness:
        h = Harness()
        h.stream.start()
        await h.stream.close()
        return h

    h = asyncio.run(scenario())
    ((url, headers),) = h.connect_calls
    assert headers == {"Authorization": f"Token {FAKE_KEY}"}
    assert FAKE_KEY not in url


def test_stream_delivers_parsed_events() -> None:
    async def scenario() -> Harness:
        h = Harness(FakeWebSocket([results("Hi", is_final=False), results("Hi there.", is_final=True, speech_final=True)]))
        h.stream.start()
        h.stream.send_audio(FRAME, 1.0)
        await asyncio.sleep(0.01)
        await h.stream.close()
        return h

    h = asyncio.run(scenario())
    transcripts = [e for e in h.events if isinstance(e, Transcript)]
    assert [(t.text, t.is_final) for t in transcripts] == [("Hi", False), ("Hi there.", True)]


def test_stream_sends_keepalive_when_no_audio_flows() -> None:
    async def scenario() -> Harness:
        h = Harness(keepalive_after_idle_s=0.02)
        h.stream.start()
        await asyncio.sleep(0.07)
        await h.stream.close()
        return h

    h = asyncio.run(scenario())
    assert h.stream.keepalives_sent >= 2
    assert h.ws.control[0] == "KeepAlive" and h.ws.control[-1] == "CloseStream"


def test_stream_no_keepalive_while_audio_is_flowing() -> None:
    async def scenario() -> Harness:
        h = Harness(keepalive_after_idle_s=0.05)
        h.stream.start()
        for _ in range(10):
            h.stream.send_audio(FRAME, 1.0)
            await asyncio.sleep(0.01)
        await h.stream.close()
        return h

    h = asyncio.run(scenario())
    assert h.stream.keepalives_sent == 0 and len(h.ws.audio) == 10


def test_connect_failure_is_reported_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    async def scenario() -> Harness:
        h = Harness(connect_error=OSError("network unreachable"))
        h.stream.start()
        await asyncio.sleep(0.01)
        h.stream.send_audio(FRAME, 1.0)  # must be a harmless no-op now
        await h.stream.close()
        return h

    with caplog.at_level(logging.ERROR, logger="voice_agent.stt"):
        h = asyncio.run(scenario())
    assert h.stream.failed
    errors = [e for e in h.events if isinstance(e, SttError)]
    assert len(errors) == 1 and "connect failed" in errors[0].message
    assert h.ws.sent == []
    assert FAKE_KEY not in caplog.text and "connect failed" in caplog.text


def test_unexpected_disconnect_mid_call_is_reported_not_raised() -> None:
    async def scenario() -> Harness:
        h = Harness()
        h.stream.start()
        h.stream.send_audio(FRAME, 1.0)
        await asyncio.sleep(0.01)
        h.ws.drop()
        await asyncio.sleep(0.01)
        for _ in range(5):
            h.stream.send_audio(FRAME, 1.0)  # call goes on; audio is dropped quietly
        await h.stream.close()
        return h

    h = asyncio.run(scenario())
    assert h.stream.failed
    assert len(h.ws.audio) == 1
    assert sum(isinstance(e, SttError) for e in h.events) == 1
    assert "CloseStream" not in h.ws.control


def test_send_failure_mid_call_is_reported_not_raised() -> None:
    async def scenario() -> Harness:
        h = Harness()
        h.stream.start()
        await asyncio.sleep(0.01)
        h.ws.fail_next_send = ConnectionError("broken pipe")
        h.stream.send_audio(FRAME, 1.0)
        await asyncio.sleep(0.01)
        await h.stream.close()
        return h

    h = asyncio.run(scenario())
    assert h.stream.failed and h.ws.closed


def test_audio_sent_before_the_connection_is_up_is_not_lost() -> None:
    async def scenario() -> Harness:
        h = Harness()
        h.stream.start()  # the connect coroutine has not run yet
        for i in range(5):
            h.stream.send_audio(bytes([i]) * 320, 1.0 + i * 0.02)
        await h.stream.close()
        return h

    h = asyncio.run(scenario())
    assert [a[0] for a in h.ws.audio] == [0, 1, 2, 3, 4]


def test_stream_that_was_never_started_is_inert() -> None:
    async def scenario() -> None:
        h = Harness()
        h.stream.send_audio(FRAME, 1.0)
        await h.stream.close()
        assert h.connect_calls == []

    asyncio.run(scenario())


# ------------------------------------------------------------------ call log + config
def test_call_recorder_writes_private_jsonl(tmp_path: Path) -> None:
    recorder = CallRecorder(tmp_path / "calls", "abc", started_wall=100.0)
    recorder.write({"type": "final", "text": "Hi, I'd like to book.", "latency_ms": 480.0}, wall=102.5)
    recorder.write({"type": "call_end", "finals": 1}, wall=103.0)
    recorder.close()
    recorder.write({"type": "ignored"})  # after close: no error
    lines = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [line["type"] for line in lines] == ["final", "call_end"]
    assert lines[0]["t"] == pytest.approx(2.5) and lines[0]["text"] == "Hi, I'd like to book."
    assert recorder.path.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "calls").stat().st_mode & 0o777 == 0o700


def test_call_recorder_survives_unwritable_directory(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    recorder = CallRecorder(blocker / "calls", "abc")
    recorder.write({"type": "final"})  # must not raise
    recorder.close()


def test_settings_defaults_and_secret_hidden_from_repr(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(f"DEEPGRAM_API_KEY={FAKE_KEY}\nSIP_1001_PASSWORD=ignored\n")
    settings = load_settings(env, environ={})
    assert settings.echo_mode is False
    assert settings.stt_active is True
    assert settings.deepgram_model == "nova-3"
    assert settings.deepgram_api_key == FAKE_KEY
    assert FAKE_KEY not in repr(settings)


def test_settings_flags(tmp_path: Path) -> None:
    missing = tmp_path / "none.env"
    assert load_settings(missing, environ={}).stt_active is False  # no key -> STT off
    on = load_settings(missing, environ={"ECHO_MODE": "true", "DEEPGRAM_API_KEY": "k", "STT_ENABLED": "false"})
    assert on.echo_mode is True and on.stt_active is False
    with pytest.raises(ValueError):
        load_settings(missing, environ={"ECHO_MODE": "maybe"})
    assert Settings().host == "127.0.0.1"
