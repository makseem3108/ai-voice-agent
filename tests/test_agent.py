"""Tests for the pacer, the conversation engine (incl. barge-in), and the
Flux / Groq / Aura clients. No network: everything external is faked."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

import httpx
import pytest

from app.agent import with_user_message
from app.audio_out import FRAME_BYTES, Pacer
from app.call_log import format_record
from app.llm_groq import GroqLLM, LLMError, LLMUsage, parse_sse_line
from app.stt_deepgram import AudioClock, SttError
from app.stt_flux import FluxInfo, FluxParams, FluxStream, TurnEvent, parse_flux_message
from app.tts_aura import AuraTTS, PhraseCache, TTSError
from tests.helpers import run
from tests.test_stt_deepgram import FakeWebSocket

FRAME = bytes(FRAME_BYTES)


# ====================================================================== pacer (mocked clock)
class FakeTime:
    """A clock that only moves when the pacer sleeps (or when we say so)."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


def run_pacer(scenario: Callable[[Pacer, FakeTime, list[float]], Any]) -> tuple[Pacer, FakeTime, list[float]]:
    async def main() -> tuple[Pacer, FakeTime, list[float]]:
        fake = FakeTime()
        sent_at: list[float] = []
        pacer = Pacer(lambda frame: sent_at.append(fake.now), clock=fake.clock, sleep=fake.sleep)
        pacer.start()
        await scenario(pacer, fake, sent_at)
        await pacer.stop()
        return pacer, fake, sent_at

    return asyncio.run(main())


def test_pacer_sends_exactly_one_frame_every_20ms() -> None:
    async def scenario(pacer: Pacer, fake: FakeTime, sent_at: list[float]) -> None:
        pacer.push(FRAME * 50)  # one second of audio, delivered all at once
        await asyncio.wait_for(pacer.drained(), 1)

    pacer, fake, sent_at = run_pacer(scenario)
    assert pacer.frames_sent == 50
    assert sent_at == pytest.approx([100.0 + i * 0.02 for i in range(50)])


def test_pacer_frames_are_320_bytes_and_keep_their_order() -> None:
    async def main() -> list[bytes]:
        fake = FakeTime()
        frames: list[bytes] = []
        pacer = Pacer(frames.append, clock=fake.clock, sleep=fake.sleep)
        pacer.start()
        audio = bytes(i % 251 for i in range(FRAME_BYTES * 3))
        for start in range(0, len(audio), 97):  # chunks that never line up with frames
            pacer.push(audio[start : start + 97])
        await asyncio.wait_for(pacer.drained(), 1)
        await pacer.stop()
        assert b"".join(frames) == audio
        return frames

    frames = asyncio.run(main())
    assert [len(f) for f in frames] == [320, 320, 320]


def test_pacer_does_not_drift_when_sending_takes_time() -> None:
    async def main() -> list[float]:
        fake = FakeTime()
        sent_at: list[float] = []

        def slow_send(frame: bytes) -> None:
            sent_at.append(fake.now)
            fake.now += 0.003  # every send "costs" 3 ms

        pacer = Pacer(slow_send, clock=fake.clock, sleep=fake.sleep)
        pacer.start()
        pacer.push(FRAME * 100)
        await asyncio.wait_for(pacer.drained(), 1)
        await pacer.stop()
        return sent_at

    sent_at = asyncio.run(main())
    # A naive "send, then sleep 20 ms" loop would be 99 * 3 ms = 297 ms late by the last frame.
    assert sent_at == pytest.approx([100.0 + i * 0.02 for i in range(100)])
    assert sent_at[-1] - sent_at[0] == pytest.approx(99 * 0.02)


def test_pacer_resyncs_after_a_stall_instead_of_bursting() -> None:
    async def main() -> tuple[list[float], int]:
        fake = FakeTime()
        sent_at: list[float] = []

        def send(frame: bytes) -> None:
            sent_at.append(fake.now)
            if len(sent_at) == 3:
                fake.now += 0.5  # the event loop froze for half a second

        pacer = Pacer(send, clock=fake.clock, sleep=fake.sleep)
        pacer.start()
        pacer.push(FRAME * 8)
        await asyncio.wait_for(pacer.drained(), 1)
        await pacer.stop()
        return sent_at, pacer.resyncs

    sent_at, resyncs = asyncio.run(main())
    assert resyncs == 1
    gaps = [b - a for a, b in zip(sent_at, sent_at[1:])]
    assert gaps[2] == pytest.approx(0.5)  # the stall itself
    assert all(g == pytest.approx(0.02) for g in gaps[3:])  # then normal pace, no catch-up burst


def test_pacer_restarts_its_schedule_after_a_pause_in_the_audio() -> None:
    async def scenario(pacer: Pacer, fake: FakeTime, sent_at: list[float]) -> None:
        pacer.push(FRAME * 2)
        await asyncio.wait_for(pacer.drained(), 1)
        fake.now += 3.0  # silence between two utterances
        pacer.push(FRAME * 2)
        await asyncio.wait_for(pacer.drained(), 1)

    _, _, sent_at = run_pacer(scenario)
    assert sent_at == pytest.approx([100.0, 100.02, 103.04, 103.06])


def test_pacer_flush_pads_the_last_partial_frame_with_silence() -> None:
    async def main() -> list[bytes]:
        fake = FakeTime()
        frames: list[bytes] = []
        pacer = Pacer(frames.append, clock=fake.clock, sleep=fake.sleep)
        pacer.start()
        pacer.push(b"\x01" * 400)
        await asyncio.sleep(0.01)
        assert len(frames) == 1  # 80 bytes are waiting for more audio
        pacer.flush()
        await asyncio.wait_for(pacer.drained(), 1)
        await pacer.stop()
        return frames

    frames = asyncio.run(main())
    assert len(frames) == 2
    assert frames[1] == b"\x01" * 80 + bytes(240)


def test_pacer_clear_drops_queued_audio_and_stops_immediately() -> None:
    async def main() -> tuple[int, int, bool]:
        fake = FakeTime()
        frames: list[bytes] = []
        pacer = Pacer(frames.append, clock=fake.clock, sleep=fake.sleep)
        pacer.start()
        pacer.push(FRAME * 100)
        while len(frames) < 5:
            await asyncio.sleep(0)
        assert pacer.speaking
        dropped = pacer.clear()
        await asyncio.wait_for(pacer.drained(), 1)
        sent = len(frames)
        await asyncio.sleep(0.01)
        still = len(frames)
        speaking = pacer.speaking
        await pacer.stop()
        assert still == sent  # nothing more after the clear
        return dropped, sent, speaking

    dropped, sent, speaking = asyncio.run(main())
    assert dropped + sent == 100 and dropped >= 90
    assert speaking is False


def test_pacer_reports_each_frame_and_empty_push_is_harmless() -> None:
    async def main() -> int:
        fake = FakeTime()
        calls = 0

        def on_frame() -> None:
            nonlocal calls
            calls += 1

        pacer = Pacer(lambda f: None, clock=fake.clock, sleep=fake.sleep, on_frame=on_frame)
        pacer.start()
        pacer.push(b"")
        assert not pacer.speaking
        await asyncio.wait_for(pacer.drained(), 1)
        pacer.push(FRAME * 3)
        await asyncio.wait_for(pacer.drained(), 1)
        await pacer.stop()
        return calls

    assert asyncio.run(main()) == 3


# ====================================================================== agent fakes
















# ====================================================================== agent behaviour














# ---------------------------------------------------------------------- eager end of turn








# ---------------------------------------------------------------------- failures










# ---------------------------------------------------------------------- ending the call






# ---------------------------------------------------------------------- helpers
def test_with_user_message_merges_consecutive_user_turns() -> None:
    history = [{"role": "assistant", "content": "Hi"}]
    one = with_user_message(history, "I want")
    two = with_user_message(one, "a widget.")
    assert two == [{"role": "assistant", "content": "Hi"}, {"role": "user", "content": "I want a widget."}]
    assert history == [{"role": "assistant", "content": "Hi"}]  # input not modified


def test_turn_and_bot_log_lines() -> None:
    turn = {"type": "turn", "turn": 3, "stt_ms": 210.4, "llm_first_token_ms": 340.0,
            "tts_first_audio_ms": 260.2, "total_ms": 830.9, "eager": True}
    assert format_record(turn) == (
        "[turn 3] stt_ms=210 llm_first_token_ms=340 tts_first_audio_ms=260 total_ms=831 eager=yes"
    )
    assert format_record({"type": "user", "text": "Hi."}) == "[user] Hi."
    assert format_record({"type": "bot", "text": "Hello."}) == "[bot]  Hello."
    assert format_record({"type": "bot", "text": "Hel", "interrupted": True}) == "[bot]  Hel (interrupted)"
    assert format_record({"type": "bot", "text": "Hi", "kind": "greeting"}) == "[bot:greeting] Hi"
    assert format_record({"type": "error", "stage": "llm", "message": "x"}) == "[error] llm: x"


# ====================================================================== Flux client
def turn_info(event: str, transcript: str = "", **extra: Any) -> str:
    body = {"type": "TurnInfo", "event": event, "turn_index": 2, "transcript": transcript,
            "end_of_turn_confidence": "0.81", "audio_window_start": "0.0", "audio_window_end": "4.8",
            "words": [], "request_id": "r", "sequence_id": 5}
    body.update(extra)
    return json.dumps(body)


def test_flux_turn_info_is_parsed_including_string_floats() -> None:
    raw = turn_info("EndOfTurn", " Book it for Tuesday. ",
                    words=[{"word": "Book", "start": 3.1, "end": 3.4}, {"word": "Tuesday.", "start": 4.0, "end": 4.42}])
    ev = parse_flux_message(raw)
    assert isinstance(ev, TurnEvent)
    assert (ev.event, ev.turn_index, ev.transcript) == ("EndOfTurn", 2, "Book it for Tuesday.")
    assert ev.eot_confidence == pytest.approx(0.81)
    assert ev.audio_window_end == pytest.approx(4.8)
    assert ev.last_word_end == pytest.approx(4.42)
    assert ev.speech_end == pytest.approx(4.42)  # last word, not window end


def test_flux_speech_end_falls_back_to_window_end_without_words() -> None:
    ev = parse_flux_message(turn_info("StartOfTurn", "Hi"))
    assert isinstance(ev, TurnEvent) and ev.last_word_end is None and ev.speech_end == pytest.approx(4.8)


def test_flux_other_messages_never_raise() -> None:
    assert parse_flux_message('{"type":"Connected","request_id":"r","sequence_id":0}') == FluxInfo("Connected")
    warning = parse_flux_message('{"type":"Warning","code":"SLOW_AUDIO","description":"audio is late"}')
    assert warning == FluxInfo("Warning", "SLOW_AUDIO audio is late")
    error = parse_flux_message('{"type":"Error","code":"INTERNAL_SERVER_ERROR","description":"oops"}')
    assert isinstance(error, SttError) and "INTERNAL_SERVER_ERROR" in error.message
    assert isinstance(parse_flux_message("not json"), SttError)
    assert isinstance(parse_flux_message("[]"), SttError)
    assert isinstance(parse_flux_message(turn_info("Update", turn_index="x")), TurnEvent)


def test_flux_url_parameters() -> None:
    url = FluxParams().url()
    assert url.startswith("wss://api.deepgram.com/v2/listen?")
    for part in ("model=flux-general-en", "encoding=linear16", "sample_rate=8000",
                 "eot_threshold=0.7", "eager_eot_threshold=0.5", "eot_timeout_ms=3000"):
        assert part in url
    assert "eager_eot_threshold" not in FluxParams(eager_eot_threshold=0).url()  # eager off
    # eager may never exceed the normal threshold (Deepgram rejects that)
    assert "eager_eot_threshold=0.6" in FluxParams(eot_threshold=0.6, eager_eot_threshold=0.9).url()
    assert "token" not in url.lower() and "key" not in url.lower()


def test_flux_stream_sends_80ms_chunks_and_closes_cleanly() -> None:
    async def scenario() -> tuple[FakeWebSocket, list[Any], FluxStream, list[tuple[str, dict[str, str]]]]:
        ws = FakeWebSocket([turn_info("EndOfTurn", "Hello.")])
        events: list[Any] = []
        calls: list[tuple[str, dict[str, str]]] = []

        async def connect(url: str, headers: dict[str, str]) -> FakeWebSocket:
            calls.append((url, headers))
            return ws

        stream = FluxStream("flux-test-key", FluxParams(), lambda ev, _t: events.append(ev), connect=connect)
        stream.start()
        for i in range(10):  # 10 frames of 20 ms
            stream.send_audio(bytes([i]) * 320, 50.0 + i * 0.02)
        await asyncio.sleep(0.01)
        await stream.close()
        return ws, events, stream, calls

    ws, events, stream, calls = run(scenario)
    assert [len(a) for a in ws.audio] == [1280, 1280, 640]  # 4 + 4 + the 2 left at hangup
    assert b"".join(ws.audio) == b"".join(bytes([i]) * 320 for i in range(10))
    assert ws.control == ["CloseStream"]
    assert stream.clock.total_seconds == pytest.approx(0.2)
    assert stream.clock.wall_time_at(0.19) == pytest.approx(50.18)  # per-frame arrival times kept
    assert any(isinstance(e, TurnEvent) and e.transcript == "Hello." for e in events)
    assert not stream.failed
    ((url, headers),) = calls
    assert headers == {"Authorization": "Token flux-test-key"} and "flux-test-key" not in url


def test_flux_disconnect_is_reported_once_and_not_raised() -> None:
    async def scenario() -> tuple[FluxStream, list[Any]]:
        ws = FakeWebSocket()
        events: list[Any] = []

        async def connect(url: str, headers: dict[str, str]) -> FakeWebSocket:
            return ws

        stream = FluxStream("k", FluxParams(), lambda ev, _t: events.append(ev), connect=connect)
        stream.start()
        await asyncio.sleep(0.01)
        ws.drop()
        await asyncio.sleep(0.01)
        stream.send_audio(FRAME, 1.0)
        await stream.close()
        return stream, events

    stream, events = run(scenario)
    assert stream.failed and sum(isinstance(e, SttError) for e in events) == 1


# ====================================================================== Groq client
def sse(*chunks: Any) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def delta(text: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


def test_parse_sse_lines() -> None:
    assert parse_sse_line("data: " + json.dumps(delta("Hi"))) == ("Hi", None, False)
    assert parse_sse_line("data: [DONE]") == ("", None, True)
    assert parse_sse_line(": keep-alive") == ("", None, False)
    assert parse_sse_line("") == ("", None, False)
    assert parse_sse_line("data: {broken") == ("", None, False)
    usage = {"prompt_tokens": 90, "completion_tokens": 12}
    assert parse_sse_line("data: " + json.dumps({"choices": [], "usage": usage})) == ("", usage, False)
    assert parse_sse_line("data: " + json.dumps({"choices": [], "x_groq": {"usage": usage}}))[1] == usage
    # reasoning text must never reach the caller
    reasoning = {"choices": [{"delta": {"reasoning": "thinking...", "content": None}}]}
    assert parse_sse_line("data: " + json.dumps(reasoning)) == ("", None, False)


def groq_with(handler: Callable[[httpx.Request], httpx.Response], model: str = "openai/gpt-oss-120b") -> GroqLLM:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GroqLLM(client, "groq-test-key", model)


def test_groq_stream_yields_text_and_counts_tokens() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        body = sse(delta("Sure. "), delta("What day?"), {"choices": [], "usage": {"prompt_tokens": 90, "completion_tokens": 12}})
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async def scenario() -> tuple[list[str], LLMUsage]:
        usage = LLMUsage()
        llm = groq_with(handler)
        out = [part async for part in llm.stream([{"role": "user", "content": "hi"}], usage)]
        return out, usage

    out, usage = run(scenario)
    assert out == ["Sure. ", "What day?"]
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, usage.requests) == (90, 12, 102, 1)
    assert seen["auth"] == "Bearer groq-test-key" and "groq-test-key" not in seen["url"]
    body = seen["body"]
    assert body["stream"] is True and body["model"] == "openai/gpt-oss-120b"
    assert body["include_reasoning"] is False and body["reasoning_effort"] == "low"
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_groq_reasoning_settings_depend_on_model() -> None:
    llm = groq_with(lambda r: httpx.Response(200), model="qwen/qwen3.8-27b")
    body = llm._body([])
    assert body["reasoning_effort"] == "none" and "include_reasoning" not in body
    assert "reasoning_effort" not in groq_with(lambda r: httpx.Response(200), model="other/model")._body([])


def test_groq_http_error_becomes_llm_error_without_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "Rate limit reached for model"}})

    async def scenario() -> str:
        with pytest.raises(LLMError) as info:
            async for _ in groq_with(handler).stream([{"role": "user", "content": "hi"}]):
                pass
        return str(info.value)

    message = run(scenario)
    assert "429" in message and "Rate limit reached" in message and "groq-test-key" not in message


def test_groq_network_error_becomes_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    async def scenario() -> None:
        with pytest.raises(LLMError, match="ConnectError"):
            async for _ in groq_with(handler).stream([]):
                pass

    run(scenario)


# ====================================================================== Aura client + phrase cache
def aura_with(handler: Callable[[httpx.Request], httpx.Response]) -> AuraTTS:
    return AuraTTS(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "dg-test-key", "aura-2-thalia-en")


def test_aura_requests_raw_8k_linear16_and_returns_audio() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=FRAME * 5)

    audio = run(lambda: aura_with(handler).synthesize("Hello there."))
    assert audio == FRAME * 5
    assert seen["params"] == {"model": "aura-2-thalia-en", "encoding": "linear16", "sample_rate": "8000", "container": "none"}
    assert seen["auth"] == "Token dg-test-key"
    assert seen["body"] == {"text": "Hello there."}


def test_aura_errors_become_tts_error() -> None:
    async def scenario() -> None:
        with pytest.raises(TTSError, match="Aura HTTP 500"):
            await aura_with(lambda r: httpx.Response(500, text="boom")).synthesize("x")

        def offline(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow")

        with pytest.raises(TTSError, match="ReadTimeout"):
            await aura_with(offline).synthesize("x")

    run(scenario)


def test_phrase_cache_synthesizes_once_then_uses_memory_and_disk(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["text"])
        return httpx.Response(200, content=FRAME * 3)

    async def scenario() -> None:
        cache = PhraseCache(aura_with(handler), tmp_path / "tts")
        assert cache.cached("Hello.") is None
        assert await cache.get("Hello.") == FRAME * 3
        assert await cache.get("Hello.") == FRAME * 3
        assert cache.cached("Hello.") == FRAME * 3
        assert calls == ["Hello."] and cache.chars_synthesized == 6
        # a new process (empty memory) finds the file and makes no request
        fresh = PhraseCache(aura_with(handler), tmp_path / "tts")
        assert await fresh.get("Hello.") == FRAME * 3
        assert calls == ["Hello."]

    run(scenario)
    (stored,) = list((tmp_path / "tts").glob("*.pcm"))
    assert stored.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "tts").stat().st_mode & 0o777 == 0o700


def test_phrase_cache_warm_never_raises(tmp_path: Path) -> None:
    async def scenario() -> PhraseCache:
        cache = PhraseCache(aura_with(lambda r: httpx.Response(503)), tmp_path / "tts")
        await cache.warm(["Hello.", "Sorry."])
        with pytest.raises(TTSError):
            await cache.get("Hello.")
        return cache

    assert run(scenario).cached("Hello.") is None


# ====================================================================== regressions found in the live self-test
def test_groq_usage_reported_twice_is_counted_once() -> None:
    numbers = {"prompt_tokens": 690, "completion_tokens": 35}

    def handler(request: httpx.Request) -> httpx.Response:
        body = sse(delta("Hi."), {"choices": [{"delta": {}}], "x_groq": {"usage": numbers}}, {"choices": [], "usage": numbers})
        return httpx.Response(200, content=body)

    async def scenario() -> LLMUsage:
        usage = LLMUsage()
        async for _ in groq_with(handler).stream([{"role": "user", "content": "hi"}], usage):
            pass
        return usage

    usage = run(scenario)
    assert (usage.prompt_tokens, usage.completion_tokens) == (690, 35)




def test_phrase_cache_concurrent_requests_synthesize_once(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=FRAME * 3)

    async def scenario() -> list[bytes]:
        cache = PhraseCache(aura_with(handler), tmp_path / "tts")
        return list(await asyncio.gather(*(cache.get("Hello.") for _ in range(5))))

    assert run(scenario) == [FRAME * 3] * 5
    assert calls == 1
