"""Chooses the speech-to-text and text-to-speech adapters a profile asks for.

A profile may say::

    stt: {provider: flux}
    tts: {provider: aura, voice: aura-2-thalia-en}

Both sections are optional; without them the profile gets Flux + Aura, the
English pipeline. Each adapter is registered here under a short name, with a
function that builds it from the profile and the settings (API keys live in
the settings, never in a profile).
"""

from __future__ import annotations

from typing import Callable

import httpx

from .config import Settings
from .profiles import Profile, ProfileError
from .speech import SpeechEventHandler, STTProvider, TTSProvider
from .stt_deepgram import AudioClock
from .stt_flux import FluxParams, FluxStream
from .tts_aura import AuraTTS

DEFAULT_STT = "flux"
DEFAULT_TTS = "aura"

SttFactory = Callable[[Profile, Settings, SpeechEventHandler, AudioClock], STTProvider]
TtsFactory = Callable[[Profile, Settings, httpx.AsyncClient], TTSProvider]


def _flux(profile: Profile, settings: Settings, on_event: SpeechEventHandler, clock: AudioClock) -> STTProvider:
    params = FluxParams(
        model=profile.stt.model or settings.flux_model,
        eot_threshold=settings.flux_eot_threshold,
        eager_eot_threshold=settings.flux_eager_eot_threshold,
        eot_timeout_ms=settings.flux_eot_timeout_ms,
        keyterms=profile.keyterms,
    )
    return FluxStream(settings.deepgram_api_key, params, on_event, clock)


def _aura(profile: Profile, settings: Settings, http: httpx.AsyncClient) -> TTSProvider:
    return AuraTTS(http, settings.deepgram_api_key, profile.tts.voice or settings.aura_model)


STT_PROVIDERS: dict[str, SttFactory] = {"flux": _flux}
TTS_PROVIDERS: dict[str, TtsFactory] = {"aura": _aura}
#: Languages an adapter can handle; None = any. Checked at start-up.
STT_LANGUAGES: dict[str, tuple[str, ...]] = {"flux": ("en",)}
TTS_LANGUAGES: dict[str, tuple[str, ...]] = {"aura": ("en",)}


def stt_name(profile: Profile) -> str:
    return profile.stt.provider or DEFAULT_STT


def tts_name(profile: Profile) -> str:
    return profile.tts.provider or DEFAULT_TTS


def check_providers(profile: Profile) -> None:
    """Fail at start-up, with a readable message, if a profile names something we cannot build."""
    problems: list[str] = []
    for kind, name, registry, languages in (
        ("stt", stt_name(profile), STT_PROVIDERS, STT_LANGUAGES),
        ("tts", tts_name(profile), TTS_PROVIDERS, TTS_LANGUAGES),
    ):
        if name not in registry:
            problems.append(f"unknown {kind} provider '{name}' (available: {', '.join(sorted(registry))})")
            continue
        supported = languages.get(name)
        language = profile.language.split("-")[0]
        if supported is not None and language not in supported:
            problems.append(
                f"{kind} provider '{name}' does not support language '{profile.language}' "
                f"(supports: {', '.join(supported)})"
            )
    if problems:
        raise ProfileError(f"profile '{profile.name}' is invalid: " + "; ".join(problems))


def make_stt(profile: Profile, settings: Settings, on_event: SpeechEventHandler, clock: AudioClock) -> STTProvider:
    """A new speech-to-text stream for one call."""
    return STT_PROVIDERS[stt_name(profile)](profile, settings, on_event, clock)


def make_tts(profile: Profile, settings: Settings, http: httpx.AsyncClient) -> TTSProvider:
    """The text-to-speech client, shared by all calls."""
    return TTS_PROVIDERS[tts_name(profile)](profile, settings, http)
