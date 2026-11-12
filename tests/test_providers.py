"""The provider seam: Flux and Aura behind STTProvider / TTSProvider, chosen by the profile.

Step 1 of phase 2c changes structure only, so most tests here pin down that
the English pipeline is built exactly as before."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.config import load_settings
from app.phrase_cache import PhraseCache
from app.profiles import ProfileError, SttConfig, TtsConfig, load_profile, parse_profile
from app.providers import STT_PROVIDERS, TTS_PROVIDERS, check_providers, make_stt, make_tts
from app.server import Services, build_services
from app.speech import SpeechEvent, STTProvider, TTSError, TTSProvider, TurnEvent
from app.stt_deepgram import AudioClock
from app.stt_flux import FluxInfo, FluxParams, FluxStream
from app.stt_flux import TurnEvent as FluxTurnEvent
from app.tts_aura import AuraTTS
from app.tts_aura import PhraseCache as PhraseCacheOldImport
from tests.helpers import PROFILE

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
ENV = {"DEEPGRAM_API_KEY": "dg-test-key", "GROQ_API_KEY": "groq-test-key"}


def settings(tmp_path: Path, **extra: str) -> Any:
    return load_settings(tmp_path / "none.env", environ={**ENV, **extra})


def ignore(event: SpeechEvent, wall: float) -> None:
    pass


# ====================================================================== the interfaces
def test_flux_and_aura_implement_the_interfaces() -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient() as http:
            assert isinstance(AuraTTS(http, "k", "aura-2-thalia-en"), TTSProvider)
        assert isinstance(FluxStream("k", FluxParams(), ignore), STTProvider)

    asyncio.run(scenario())


def test_old_import_paths_still_work() -> None:
    assert FluxTurnEvent is TurnEvent and PhraseCacheOldImport is PhraseCache
    assert FluxInfo("Connected").type == "Connected"


def test_flux_reports_eager_support_from_its_parameters() -> None:
    async def scenario() -> None:
        assert FluxStream("k", FluxParams(), ignore).supports_eager is True
        assert FluxStream("k", FluxParams(eager_eot_threshold=0), ignore).supports_eager is False
        stream = FluxStream("k", FluxParams(model="flux-general-en"), ignore)
        assert (stream.name, stream.model, stream.failed) == ("flux", "flux-general-en", False)

    asyncio.run(scenario())


# ====================================================================== the factory: defaults = today's English pipeline
def test_shipped_profiles_need_no_changes_and_default_to_english() -> None:
    for name in ("lead_qualification", "clinic", "travel"):
        profile = load_profile(name, PROFILES_DIR)
        assert profile.language == "en" and profile.stt == SttConfig() and profile.tts == TtsConfig()
        check_providers(profile)


def test_default_profile_gets_flux_with_exactly_the_same_request(tmp_path: Path) -> None:
    async def scenario() -> None:
        s = settings(tmp_path, FLUX_EOT_THRESHOLD="0.8", FLUX_EAGER_EOT_THRESHOLD="0.6", FLUX_EOT_TIMEOUT_MS="2500")
        profile = dataclasses.replace(PROFILE, keyterms=("solar", "electricity bill"))
        clock = AudioClock()
        stream = make_stt(profile, s, ignore, clock)
        assert isinstance(stream, FluxStream) and stream.clock is clock
        by_hand = FluxParams(
            model=s.flux_model, eot_threshold=0.8, eager_eot_threshold=0.6, eot_timeout_ms=2500,
            keyterms=("solar", "electricity bill"),
        )
        assert stream._params == by_hand and stream._params.url() == by_hand.url()
        assert stream._api_key == "dg-test-key"

    asyncio.run(scenario())


def test_default_profile_gets_aura_with_the_configured_voice(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient() as http:
            tts = make_tts(PROFILE, settings(tmp_path), http)
            assert isinstance(tts, AuraTTS)
            assert (tts.name, tts.voice, tts.cache_key) == ("Aura", "aura-2-thalia-en", "aura-2-thalia-en")
            assert tts.keep_warm_url == "https://api.deepgram.com/"
            other = make_tts(PROFILE, settings(tmp_path, AURA_MODEL="aura-2-apollo-en"), http)
            assert other.voice == "aura-2-apollo-en"

    asyncio.run(scenario())


def test_profile_can_override_model_and_voice(tmp_path: Path) -> None:
    async def scenario() -> None:
        profile = dataclasses.replace(
            PROFILE, stt=SttConfig(provider="flux", model="flux-general-multi"), tts=TtsConfig(provider="aura", voice="aura-2-luna-en")
        )
        async with httpx.AsyncClient() as http:
            assert make_tts(profile, settings(tmp_path), http).voice == "aura-2-luna-en"
        assert make_stt(profile, settings(tmp_path), ignore, AudioClock()).model == "flux-general-multi"

    asyncio.run(scenario())


def test_unknown_provider_or_unsupported_language_stops_at_start_up() -> None:
    with pytest.raises(ProfileError, match=r"unknown stt provider 'whisper' \(available: flux\)"):
        check_providers(dataclasses.replace(PROFILE, stt=SttConfig(provider="whisper")))
    with pytest.raises(ProfileError, match=r"unknown tts provider 'robot' \(available: aura\)"):
        check_providers(dataclasses.replace(PROFILE, tts=TtsConfig(provider="robot")))
    with pytest.raises(ProfileError) as info:
        check_providers(dataclasses.replace(PROFILE, language="he"))
    message = str(info.value)
    assert "stt provider 'flux' does not support language 'he'" in message
    assert "tts provider 'aura' does not support language 'he'" in message
    check_providers(dataclasses.replace(PROFILE, language="en-US"))
    assert set(STT_PROVIDERS) == {"flux"} and set(TTS_PROVIDERS) == {"aura"}


def test_build_services_uses_the_factory(tmp_path: Path) -> None:
    async def scenario() -> Services:
        s = dataclasses.replace(settings(tmp_path), profiles_dir=PROFILES_DIR, tts_cache_dir=tmp_path / "tts")
        services = build_services(s)
        assert services.http is not None
        await services.http.aclose()
        return services

    services = asyncio.run(scenario())
    assert isinstance(services.tts, AuraTTS) and isinstance(services.phrases, PhraseCache)
    assert services.agent_ready and services.mode == "agent"


# ====================================================================== profile fields
def profile_data(**extra: Any) -> dict[str, Any]:
    return {
        "business_name": "Acme",
        "agent_role": "the receptionist",
        "goal": "Take a message.",
        "greeting": "Hello.",
        "slots": [{"name": "ok", "type": "yes_no", "question": "Okay?"}],
        "knowledge": "Nothing.",
        "tone": "friendly",
        "closing_qualified": "Bye!",
        "closing_not_fit": "Sorry, bye!",
        **extra,
    }


def test_profile_reads_language_stt_and_tts() -> None:
    profile = parse_profile(
        "t",
        profile_data(language="ar-JO", stt={"provider": "flux", "model": "m", "language": "ar"}, tts={"provider": "aura", "voice": "v"}),
    )
    assert profile.language == "ar-JO"
    assert profile.stt == SttConfig("flux", "m", "ar") and profile.tts == TtsConfig("aura", "v")
    assert parse_profile("t", profile_data(tts={"voice": "v"})).tts == TtsConfig("", "v")


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"language": "Hebrew"}, "'language' must be a language code"),
        ({"language": 5}, "'language' must be a language code"),
        ({"stt": "flux"}, "'stt' must be a mapping"),
        ({"stt": {"provider": "flux", "speed": 2}}, "'stt': unknown field(s): speed"),
        ({"stt": {"provider": "Flux!"}}, "'stt.provider' must be an adapter name"),
        ({"tts": {"voice": ""}}, "'tts.voice' must be short non-empty text"),
        ({"tts": {"provider": "aura", "model": "x"}}, "'tts': unknown field(s): model"),
    ],
)
def test_profile_rejects_bad_language_and_provider_sections(extra: dict[str, Any], message: str) -> None:
    with pytest.raises(ProfileError) as info:
        parse_profile("t", profile_data(**extra))
    assert message in str(info.value)


# ====================================================================== phrase cache and keep-warm
class OneVoice:
    name = "Robo"
    keep_warm_url = None

    def __init__(self, voice: str, audio: bytes = b"\x01\x02" * 80) -> None:
        self.voice = self.cache_key = voice
        self.audio = audio
        self.requests = 0

    async def stream(self, text: str):  # type: ignore[no-untyped-def]
        yield self.audio

    async def synthesize(self, text: str) -> bytes:
        self.requests += 1
        return self.audio


def test_existing_aura_cache_files_stay_valid(tmp_path: Path) -> None:
    """The file name is still sha256("<aura model>|8000|<text>"), as before the refactor."""

    async def scenario() -> list[str]:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return httpx.Response(200, content=b"\x00\x01" * 160)

        tts = AuraTTS(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "k", "aura-2-thalia-en")
        old_name = hashlib.sha256("aura-2-thalia-en|8000|Hello there.".encode()).hexdigest()[:32] + ".pcm"
        (tmp_path / old_name).write_bytes(b"\x07\x07" * 160)
        assert await PhraseCache(tts, tmp_path).get("Hello there.") == b"\x07\x07" * 160
        return calls

    assert asyncio.run(scenario()) == []  # served from the old file, no request


def test_each_voice_has_its_own_cached_audio(tmp_path: Path) -> None:
    async def scenario() -> tuple[OneVoice, OneVoice]:
        hila, sana = OneVoice("he-IL-HilaNeural", b"\x01\x01" * 80), OneVoice("ar-JO-SanaNeural", b"\x02\x02" * 80)
        assert await PhraseCache(hila, tmp_path).get("Hello.") == hila.audio
        assert await PhraseCache(sana, tmp_path).get("Hello.") == sana.audio  # not Hila's recording
        assert await PhraseCache(hila, tmp_path).get("Hello.") == hila.audio  # from disk
        return hila, sana

    hila, sana = asyncio.run(scenario())
    assert (hila.requests, sana.requests) == (1, 1) and len(list(tmp_path.glob("*.pcm"))) == 2


def test_phrase_cache_error_names_the_provider(tmp_path: Path) -> None:
    async def scenario() -> None:
        with pytest.raises(TTSError, match="Robo returned no audio"):
            await PhraseCache(OneVoice("v", b""), tmp_path).get("Hello.")

    asyncio.run(scenario())


def _warm_urls(tmp_path: Path, tts: Any) -> list[str]:
    async def scenario() -> list[str]:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200)

        services = Services(settings(tmp_path), http=httpx.AsyncClient(transport=httpx.MockTransport(handler)), tts=tts)
        task = asyncio.get_running_loop().create_task(services.keep_warm())
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return seen

    return asyncio.run(scenario())


def test_keep_warm_touches_only_what_the_tts_provider_asks_for(tmp_path: Path) -> None:
    async def aura() -> AuraTTS:
        return AuraTTS(httpx.AsyncClient(), "k")

    assert _warm_urls(tmp_path, asyncio.run(aura())) == ["https://api.deepgram.com/", "https://api.groq.com/"]
    assert _warm_urls(tmp_path, OneVoice("v")) == ["https://api.groq.com/"]  # a provider that must not be pinged
