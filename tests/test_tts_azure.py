"""Azure text-to-speech adapter, built for the free tier: mocked HTTP, no network."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
import wave
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

import httpx
import pytest

from app.agent import APOLOGY_TEXT, ONE_MOMENT_TEXT, fixed_phrases
from app.call_log import format_record
from app.config import load_settings
from app.phrase_cache import PhraseCache
from app.profiles import ProfileError, TtsConfig
from app.providers import check_providers, make_tts
from app.server import tts_usage_record
from app.speech import TTSError, TTSProvider, TTSRateLimited
from app.tts_aura import AuraTTS
from app.tts_azure import (
    FREE_TIER_CHARS_PER_MONTH,
    AzureTTS,
    ConnectionWatch,
    RateLimiter,
    UsageMeter,
    build_ssml,
    default_voice,
    endpoint,
)
from tests.helpers import FRAME, PROFILE, FakeLLM, FakePhrases, FakeTTS, Rig, reply, run, until

HILA = "he-IL-HilaNeural"
HEBREW = "שלום, במה אפשר לעזור?"
PCM = b"\x01\x02" * 400


def azure(handler: Callable[[httpx.Request], httpx.Response], voice: str = HILA, **kwargs: Any) -> AzureTTS:
    return AzureTTS(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "azure-test-key", "northeurope", voice, **kwargs)


def collect(tts: AzureTTS, text: str = HEBREW) -> bytes:
    return asyncio.run(tts.synthesize(text))


# ====================================================================== the request
def test_request_asks_for_phone_audio_and_carries_the_key_only_in_a_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=PCM)

    tts = azure(handler)
    assert collect(tts) == PCM
    (request,) = seen
    assert str(request.url) == "https://northeurope.tts.speech.microsoft.com/cognitiveservices/v1" == endpoint("northeurope")
    assert request.headers["X-Microsoft-OutputFormat"] == "raw-8khz-16bit-mono-pcm"  # no resampling needed
    assert request.headers["Content-Type"] == "application/ssml+xml"
    assert request.headers["Ocp-Apim-Subscription-Key"] == "azure-test-key"
    assert "azure-test-key" not in str(request.url) and "azure-test-key" not in request.content.decode()
    assert request.content.decode() == f"<speak version='1.0' xml:lang='he-IL'><voice name='{HILA}'>{HEBREW}</voice></speak>"


def test_ssml_escapes_the_text_so_it_cannot_add_markup() -> None:
    ssml = build_ssml("ar-JO-SanaNeural", "5 < 6 & <voice name='x'>hi</voice>")
    assert ssml.count("<voice") == 1 and "&lt;voice name='x'&gt;" in ssml and "5 &lt; 6 &amp;" in ssml
    assert ssml.startswith("<speak version='1.0' xml:lang='ar-JO'><voice name='ar-JO-SanaNeural'>")


def test_it_is_a_tts_provider_built_for_the_free_tier() -> None:
    tts = azure(lambda r: httpx.Response(200, content=PCM))
    assert isinstance(tts, TTSProvider)
    assert tts.name == "Azure" and tts.voice == HILA and tts.cache_key == f"azure:{HILA}"
    assert tts.keep_warm_url is None  # never pinged: every request counts on F0
    assert tts.one_request_per_reply is True
    assert default_voice("he") == HILA and default_voice("ar-JO") == "ar-JO-SanaNeural" and default_voice("xx") is None


def test_audio_is_streamed_chunk_by_chunk() -> None:
    class Slow(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            for part in (b"\x01\x01", b"", b"\x02\x02", b"\x03\x03"):
                yield part

    async def scenario() -> list[bytes]:
        tts = azure(lambda r: httpx.Response(200, stream=Slow()))
        return [chunk async for chunk in tts.stream(HEBREW)]

    assert asyncio.run(scenario()) == [b"\x01\x01", b"\x02\x02", b"\x03\x03"]  # empty chunks are skipped


def test_stopping_early_closes_the_request_so_synthesis_ends() -> None:
    """Barge-in: the agent stops reading; the HTTP response must be closed, not read to the end."""
    state = {"closed": False, "served": 0}

    class Endless(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            while True:
                state["served"] += 1
                yield FRAME
                await asyncio.sleep(0)

        async def aclose(self) -> None:
            state["closed"] = True

    async def scenario() -> None:
        tts = azure(lambda r: httpx.Response(200, stream=Endless()))

        async def speak() -> None:
            async for _ in tts.stream(HEBREW):
                await asyncio.sleep(0.001)

        task = asyncio.get_running_loop().create_task(speak())
        await asyncio.sleep(0.03)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert state["closed"] is True and 0 < state["served"] < 1000


# ====================================================================== errors
def test_http_429_becomes_rate_limited_with_the_wait_azure_asks_for() -> None:
    tts = azure(lambda r: httpx.Response(429, headers={"Retry-After": "7"}, text="too many"))
    with pytest.raises(TTSRateLimited) as info:
        collect(tts)
    assert info.value.retry_after_s == 7.0 and isinstance(info.value, TTSError)
    with pytest.raises(TTSRateLimited) as plain:
        collect(azure(lambda r: httpx.Response(429)))
    assert plain.value.retry_after_s == 3.0  # no header: a sensible default


@pytest.mark.parametrize("status", [400, 401, 403, 500, 503])
def test_other_http_errors_are_plain_tts_errors_without_the_key(status: int) -> None:
    with pytest.raises(TTSError) as info:
        collect(azure(lambda r: httpx.Response(status, text="azure-test-key leaked?")))
    assert str(info.value) == f"Azure HTTP {status}" and not isinstance(info.value, TTSRateLimited)


def test_network_failure_is_a_tts_error() -> None:
    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow")

    with pytest.raises(TTSError, match="Azure request failed: ConnectTimeout"):
        collect(azure(offline))


# ====================================================================== our own 20-per-minute limiter
class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_rate_limiter_is_a_sliding_window() -> None:
    clock = Clock()
    limiter = RateLimiter(limit=3, window_s=60.0, now=clock)
    assert [limiter.reserve() for _ in range(3)] == [0.0, 0.0, 0.0] and limiter.used == 3
    assert limiter.reserve() == pytest.approx(60.0)  # refused: nothing is taken
    clock.t += 20
    assert limiter.reserve() == pytest.approx(40.0) and limiter.used == 3
    clock.t += 40.5  # the first three have left the window
    assert limiter.reserve() == 0.0 and limiter.used == 1
    assert RateLimiter().limit == 20  # the free tier's limit is the default


def test_the_21st_request_of_a_minute_is_refused_before_it_is_sent() -> None:
    sent = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        sent += 1
        return httpx.Response(200, content=PCM)

    clock = Clock()

    async def scenario() -> float:
        tts = azure(handler, limiter=RateLimiter(now=clock))
        for _ in range(20):
            await tts.synthesize("x")
            clock.t += 1
        with pytest.raises(TTSRateLimited) as info:
            await tts.synthesize("x")
        return info.value.retry_after_s

    wait = asyncio.run(scenario())
    assert sent == 20  # the 21st never left the building
    assert wait == pytest.approx(40.0)  # the first request leaves the window 60 s after it was sent


# ====================================================================== usage / monthly quota
def test_usage_is_counted_per_month_on_disk(tmp_path: Path) -> None:
    clock = Clock()
    clock.t = 1794000000.0  # 2026-11
    meter = UsageMeter(tmp_path, now=clock)
    tts = azure(lambda r: httpx.Response(200, content=PCM), usage=meter)
    collect(tts, "שלום")
    collect(tts, "מה שלומך?")
    assert (meter.requests, meter.chars) == (2, 4 + 9)
    assert meter.month() == {"month": "2026-11", "requests": 2, "chars": 13}
    assert json.loads((tmp_path / "azure_tts_2026-11.json").read_text()) == {"month": "2026-11", "requests": 2, "chars": 13}
    again = UsageMeter(tmp_path, now=clock)  # a restart (or the CLI) continues the same month
    again.add(7)
    assert again.month()["chars"] == 20 and again.chars == 7
    clock.t += 40 * 86400  # next month starts from zero
    assert again.month() == {"month": "2026-12", "requests": 0, "chars": 0}


def test_refused_requests_do_not_count_as_characters_used(tmp_path: Path) -> None:
    meter = UsageMeter(tmp_path)
    with pytest.raises(TTSRateLimited):
        collect(azure(lambda r: httpx.Response(429), usage=meter))
    with pytest.raises(TTSError):
        collect(azure(lambda r: httpx.Response(500), usage=meter))
    assert meter.chars == 0 and meter.requests == 0


def test_usage_meter_survives_a_broken_file_and_works_without_a_directory(tmp_path: Path) -> None:
    meter = UsageMeter(tmp_path)
    (tmp_path / f"azure_tts_{meter.month()['month']}.json").write_text("not json")
    meter.add(5)
    assert meter.month()["chars"] == 5
    memory_only = UsageMeter()
    memory_only.add(3)
    assert memory_only.chars == 3 and memory_only.month()["chars"] == 0


def test_usage_record_for_the_call_log(tmp_path: Path) -> None:
    meter = UsageMeter(tmp_path)
    tts = azure(lambda r: httpx.Response(200, content=PCM), usage=meter)
    collect(tts, "x" * 120)
    record = tts_usage_record(tts, 120)
    assert record is not None
    assert record["provider"] == "Azure" and record["voice"] == HILA and record["call_chars"] == 120
    assert record["month_chars"] == 120 and record["month_requests"] == 1 and record["month_limit"] == FREE_TIER_CHARS_PER_MONTH == 500_000
    line = format_record(record)
    assert line is not None and line.startswith("[tts_usage] Azure: this call 120 characters; month ") and "120 of 500000 characters, 1 requests" in line

    async def aura() -> AuraTTS:
        return AuraTTS(httpx.AsyncClient(), "k")

    assert tts_usage_record(asyncio.run(aura()), 50) is None and tts_usage_record(None, 0) is None  # Aura has no monthly allowance


# ====================================================================== connection reuse
def test_connection_watch_tells_a_fresh_connection_from_a_reused_one() -> None:
    async def scenario() -> tuple[Optional[float], Optional[float]]:
        clock = Clock()
        fresh = ConnectionWatch(now=clock)
        await fresh("connection.connect_tcp.started", {})
        clock.t += 0.05
        await fresh("connection.connect_tcp.complete", {})
        await fresh("connection.start_tls.started", {})
        clock.t += 0.12
        await fresh("connection.start_tls.complete", {})
        await fresh("http11.send_request_headers.started", {})
        reused = ConnectionWatch(now=clock)
        await reused("http11.send_request_headers.started", {})
        await reused("http11.receive_response_headers.complete", {})
        return fresh.connect_ms, reused.connect_ms

    fresh_ms, reused_ms = asyncio.run(scenario())
    assert fresh_ms == pytest.approx(170.0) and reused_ms is None


# ====================================================================== fixed phrases cost nothing
def test_fixed_phrases_are_cached_per_voice_so_calls_need_no_request(tmp_path: Path) -> None:
    sent = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        sent += 1
        return httpx.Response(200, content=PCM)

    async def scenario() -> None:
        texts = fixed_phrases(PROFILE)
        await PhraseCache(azure(handler), tmp_path).warm(texts)
        assert sent == len(texts)  # once, ever
        restarted = PhraseCache(azure(handler), tmp_path)
        await restarted.warm(texts)
        assert sent == len(texts) and restarted.cached(PROFILE.greeting) == PCM  # from disk: zero requests
        await PhraseCache(azure(handler, voice="he-IL-AvriNeural"), tmp_path).get(PROFILE.greeting)
        assert sent == len(texts) + 1  # another voice has its own recording

    asyncio.run(scenario())
    assert ONE_MOMENT_TEXT in fixed_phrases(PROFILE) and APOLOGY_TEXT in fixed_phrases(PROFILE)


# ====================================================================== factory
def test_factory_builds_azure_from_profile_and_settings(tmp_path: Path) -> None:
    async def scenario() -> None:
        s = load_settings(tmp_path / "none.env", environ={"AZURE_SPEECH_KEY": "k", "AZURE_SPEECH_REGION": "NorthEurope", "USAGE_DIR": str(tmp_path)})
        async with httpx.AsyncClient() as http:
            hebrew = dataclasses.replace(PROFILE, language="he", tts=TtsConfig(provider="azure"))
            tts = make_tts(hebrew, s, http)
            assert isinstance(tts, AzureTTS) and tts.voice == HILA and tts.limiter.limit == 20
            assert tts._url == endpoint("northeurope")
            arabic = dataclasses.replace(PROFILE, language="ar", tts=TtsConfig(provider="azure"))
            assert make_tts(arabic, s, http).voice == "ar-JO-SanaNeural"
            chosen = dataclasses.replace(arabic, tts=TtsConfig(provider="azure", voice="ar-LB-LaylaNeural"))
            assert make_tts(chosen, s, http).voice == "ar-LB-LaylaNeural"
            slower = load_settings(tmp_path / "none.env", environ={"AZURE_TTS_MAX_PER_MINUTE": "10"})
            assert make_tts(hebrew, slower, http).limiter.limit == 10

    asyncio.run(scenario())


def test_azure_profile_without_key_or_voice_stops_at_start_up(tmp_path: Path) -> None:
    hebrew = dataclasses.replace(PROFILE, language="he", stt=PROFILE.stt, tts=TtsConfig(provider="azure"))
    no_key = load_settings(tmp_path / "none.env", environ={})
    with pytest.raises(ProfileError, match="needs AZURE_SPEECH_KEY and AZURE_SPEECH_REGION"):
        check_providers(hebrew, no_key)
    with pytest.raises(ProfileError, match="no default voice for language 'fr'"):
        check_providers(dataclasses.replace(hebrew, language="fr"))
    defaults = load_settings(tmp_path / "none.env", environ={})
    assert defaults.azure_tts_max_per_minute == 20 and defaults.tmp_audio_dir.name == "tmp" and defaults.usage_dir.name == "usage"
    assert "azure_speech_key" not in repr(dataclasses.replace(defaults, azure_speech_key="secret-value")) or "secret-value" not in repr(
        dataclasses.replace(defaults, azure_speech_key="secret-value")
    )


# ====================================================================== the agent with a one-request-per-reply voice
class WholeReplyTTS(FakeTTS):
    one_request_per_reply = True

    def __init__(self, limited: int = 0, retry_after_s: float = 0.05) -> None:
        super().__init__(frames_per_sentence=2)
        self.limited = limited  # how many requests are refused first
        self.retry_after_s = retry_after_s
        self.attempts = 0

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        self.attempts += 1
        if self.attempts <= self.limited:
            raise TTSRateLimited("Azure HTTP 429", self.retry_after_s)
        async for chunk in super().stream(text):
            yield chunk


def test_whole_reply_goes_out_as_one_request() -> None:
    async def scenario() -> Rig:
        tts = WholeReplyTTS()
        rig = Rig(FakeLLM([reply("Thanks for that. We are open every day. Do you own the house?")]), tts)
        rig.pacer.start()
        await rig.say("Hi.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.tts.spoken == ["Thanks for that. We are open every day. Do you own the house?"]  # one request, three sentences
    assert rig.agent.tts_chars == len(rig.tts.spoken[0]) and rig.of("turn")[0]["tts_first_audio_ms"] is not None


def test_per_sentence_voices_are_unchanged() -> None:
    async def scenario() -> Rig:
        rig = Rig(FakeLLM([reply("Thanks for that. Do you own the house?")]))
        rig.pacer.start()
        await rig.say("Hi.")
        await rig.agent.close()
        return rig

    assert run(scenario).tts.spoken == ["Thanks for that.", "Do you own the house?"]  # Aura keeps streaming sentence by sentence


def test_rate_limit_says_one_moment_and_retries_once() -> None:
    async def scenario() -> Rig:
        tts = WholeReplyTTS(limited=1)
        rig = Rig(FakeLLM([reply("Do you own the house?")]), tts)
        rig.pacer.start()
        await rig.say("Hi.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.tts.attempts == 2 and rig.tts.spoken == ["Do you own the house?"]
    assert ONE_MOMENT_TEXT in rig.phrases.served  # cached phrase: no request needed to say it
    assert rig.bot_lines() == ["Do you own the house?"] and rig.agent.errors == 0
    (record,) = rig.of("tts_rate_limited")
    assert record == {"type": "tts_rate_limited", "retry_after_s": 0.1, "waiting_s": 0.1} or record["waiting_s"] <= 0.1
    assert rig.agent.summary()["tts_rate_limited"] == 1 and rig.agent.tts_chars == len("Do you own the house?")  # counted once
    assert (format_record(record) or "").startswith("[tts_rate_limited]")


def test_second_refusal_is_an_error_and_the_cached_apology_is_played() -> None:
    async def scenario() -> Rig:
        tts = WholeReplyTTS(limited=2)
        rig = Rig(FakeLLM([reply("Do you own the house?")]), tts)
        rig.pacer.start()
        await rig.say("Hi.")
        await rig.agent.close()
        return rig

    rig = run(scenario)
    assert rig.tts.attempts == 2  # exactly one retry
    assert rig.agent.errors == 1 and rig.of("error")[0]["stage"] == "tts"
    assert rig.of("bot")[-1]["kind"] == "apology"


def test_long_waits_are_capped_and_barge_in_cancels_the_retry() -> None:
    async def scenario() -> Rig:
        tts = WholeReplyTTS(limited=1, retry_after_s=40.0)  # our own limiter can ask for most of a minute
        rig = Rig(FakeLLM([reply("Do you own the house?"), reply("Sure.")]), tts)
        rig.pacer.start()
        rig.event("StartOfTurn", "Hi")
        rig.event("EndOfTurn", "Hi.")
        await until(lambda: bool(rig.of("tts_rate_limited")))
        assert rig.of("tts_rate_limited")[0]["waiting_s"] == 4.0  # never 40 s of silence
        rig.event("StartOfTurn", "Hello")  # the caller speaks during the wait
        await asyncio.sleep(0.05)
        assert rig.agent._job is None and tts.attempts == 1  # the retry was dropped with the reply
        await rig.agent.close()
        return rig

    run(scenario)


# ====================================================================== the listen CLI
def test_cli_writes_a_phone_quality_wav_and_reports_timing(tmp_path: Path) -> None:
    import scripts.say as say

    async def scenario() -> dict[str, Any]:
        tts = azure(lambda r: httpx.Response(200, content=PCM), voice="ar-JO-SanaNeural")
        return await say.synthesize_to_file(tts, say.SENTENCES["ar"], tmp_path / "tmp")

    result = asyncio.run(scenario())
    path: Path = result["path"]
    assert path.parent == tmp_path / "tmp" and path.name.endswith("-ar-JO-SanaNeural.wav")
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    with wave.open(str(path), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 8000)
        assert wav.readframes(wav.getnframes()) == PCM
    assert result["chars"] == len(say.SENTENCES["ar"]) and result["audio_s"] == pytest.approx(0.05)
    assert result["first_audio_ms"] is not None and result["total_ms"] >= result["first_audio_ms"]
    assert [v for _, v in say.SUGGESTED] == ["he-IL-HilaNeural", "he-IL-AvriNeural", "ar-JO-SanaNeural", "ar-LB-LaylaNeural", "ar-JO-TaimNeural"]


def test_cli_fails_cleanly_when_azure_returns_nothing(tmp_path: Path) -> None:
    import scripts.say as say

    async def scenario() -> None:
        with pytest.raises(TTSError, match="no audio"):
            await say.synthesize_to_file(azure(lambda r: httpx.Response(200, content=b"")), "x", tmp_path)

    asyncio.run(scenario())
    assert list(tmp_path.glob("*.wav")) == []


def test_test_audio_is_deleted_after_seven_days(tmp_path: Path) -> None:
    import os

    from app.recording import cleanup_old_recordings

    old, fresh = tmp_path / "20261101-000000-he-IL-HilaNeural.wav", tmp_path / "20261112-000000-he-IL-HilaNeural.wav"
    old.write_bytes(b"x")
    fresh.write_bytes(b"x")
    os.utime(old, (time.time() - 8 * 86400, time.time() - 8 * 86400))
    assert cleanup_old_recordings(tmp_path) == 1 and fresh.exists() and not old.exists()
