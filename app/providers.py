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

from typing import Callable, Optional

import httpx

from .config import Settings
from .profiles import Profile, ProfileError
from .speech import SpeechEventHandler, STTProvider, TTSProvider
from .stt_deepgram import AudioClock
from .stt_deepgram import DeepgramParams
from .stt_flux import FluxParams, FluxStream
from .stt_nova import NovaStream
from .tts_aura import AuraTTS
from .tts_azure import AzureTTS, RateLimiter, UsageMeter, default_voice

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


def stt_language(profile: Profile) -> str:
    """The language code sent to speech-to-text: the profile's, unless stt.language overrides it."""
    return profile.stt.language or profile.language


def _nova(profile: Profile, settings: Settings, on_event: SpeechEventHandler, clock: AudioClock) -> STTProvider:
    params = DeepgramParams(
        model=profile.stt.model or "nova-3",
        language=stt_language(profile),
        endpointing_ms=settings.nova_endpointing_ms,
        utterance_end_ms=settings.nova_utterance_end_ms,
        keyterms=profile.keyterms,
        language_hints=profile.stt.language_hints,
        smart_format=False,  # keep numbers as words; app/amounts.py reads them in he / ar / en
    )
    return NovaStream(
        settings.deepgram_api_key, params, on_event, clock, turn_timeout_s=settings.nova_turn_timeout_ms / 1000.0
    )


def describe_stt(profile: Profile, settings: Settings) -> str:
    """Short label for the start-up log line, e.g. ``flux-general-en`` or ``nova-3/he``."""
    if stt_name(profile) == "nova3":
        hints = f"[{','.join(profile.stt.language_hints)}]" if profile.stt.language_hints else ""
        return f"{profile.stt.model or 'nova-3'}/{stt_language(profile)}{hints}"
    return profile.stt.model or settings.flux_model


def stt_eager(profile: Profile, settings: Settings) -> bool:
    """Will this profile's speech-to-text give the agent speculative head starts?"""
    return stt_name(profile) == "flux" and settings.flux_eager_eot_threshold > 0


def _aura(profile: Profile, settings: Settings, http: httpx.AsyncClient) -> TTSProvider:
    return AuraTTS(http, settings.deepgram_api_key, profile.tts.voice or settings.aura_model)


#: Azure counts requests per subscription, not per voice: every Azure voice in
#: this process shares one limiter and one usage meter.
_AZURE_SHARED: dict[tuple[str, int, str], tuple[RateLimiter, UsageMeter]] = {}


def _azure(profile: Profile, settings: Settings, http: httpx.AsyncClient) -> TTSProvider:
    voice = profile.tts.voice or default_voice(profile.language) or ""
    key = (settings.azure_speech_region, settings.azure_tts_max_per_minute, str(settings.usage_dir))
    if key not in _AZURE_SHARED:
        _AZURE_SHARED[key] = (RateLimiter(settings.azure_tts_max_per_minute), UsageMeter(settings.usage_dir))
    limiter, usage = _AZURE_SHARED[key]
    return AzureTTS(http, settings.azure_speech_key, settings.azure_speech_region, voice, limiter=limiter, usage=usage)


STT_PROVIDERS: dict[str, SttFactory] = {"flux": _flux, "nova3": _nova}
TTS_PROVIDERS: dict[str, TtsFactory] = {"aura": _aura, "azure": _azure}
#: Languages an adapter can handle; a missing entry = any. Checked at start-up.
STT_LANGUAGES: dict[str, tuple[str, ...]] = {"flux": ("en",)}
TTS_LANGUAGES: dict[str, tuple[str, ...]] = {"aura": ("en",)}


def stt_name(profile: Profile) -> str:
    return profile.stt.provider or DEFAULT_STT


def tts_name(profile: Profile) -> str:
    return profile.tts.provider or DEFAULT_TTS


def check_providers(profile: Profile, settings: Optional[Settings] = None) -> None:
    """Fail at start-up, with a readable message, if a profile names something we cannot build.

    With ``settings`` it also checks that the chosen services are configured
    (keys are only tested for presence, never printed).
    """
    problems: list[str] = []
    if tts_name(profile) == "azure":
        if not (profile.tts.voice or default_voice(profile.language)):
            problems.append(f"tts provider 'azure' has no default voice for language '{profile.language}'; set tts.voice")
        if settings is not None and not (settings.azure_speech_key and settings.azure_speech_region):
            problems.append("tts provider 'azure' needs AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env")
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
