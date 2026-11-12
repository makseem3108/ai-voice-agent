"""Runtime configuration, read from the process environment and ``.env``.

Real environment variables win over ``.env``. Only the keys this app needs are
read; the file may hold other secrets (e.g. SIP credentials) that are ignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

#: The AudioSocket protocol has no authentication or encryption, so the server
#: is loopback-only by design. This is intentionally not configurable.
BIND_HOST = "127.0.0.1"


@dataclass(frozen=True)
class Settings:
    host: str = BIND_HOST
    port: int = 9092
    log_level: str = "INFO"

    #: Debug aid from phase 1: play the caller's own audio back after a delay.
    echo_mode: bool = False
    echo_delay_ms: int = 1000

    #: Speech-to-text. ``repr=False`` keeps the key out of logs and tracebacks.
    deepgram_api_key: str = field(default="", repr=False)
    stt_enabled: bool = True
    deepgram_model: str = "nova-3"
    deepgram_language: str = "en"
    deepgram_endpointing_ms: int = 400
    deepgram_utterance_end_ms: int = 1000

    #: Conversational agent (phase 2b). Needs both API keys; otherwise the
    #: server falls back to transcription only (phase 2a behaviour).
    agent_enabled: bool = True
    active_profile: str = "lead_qualification"
    profiles_dir: Path = PROJECT_ROOT / "profiles"
    groq_api_key: str = field(default="", repr=False)
    groq_model: str = "openai/gpt-oss-120b"
    #: Eager head starts are skipped while less than this share of the
    #: per-minute token allowance is left (0 = never skip).
    groq_eager_min_budget: float = 0.4
    aura_model: str = "aura-2-thalia-en"
    flux_model: str = "flux-general-en"
    flux_eot_threshold: float = 0.7
    #: 0 switches the eager head start off (fewer LLM requests, slightly slower replies).
    flux_eager_eot_threshold: float = 0.5
    flux_eot_timeout_ms: int = 3000

    #: Fast barge-in: local voice detection pauses the bot before STT confirms.
    barge_vad_enabled: bool = True
    barge_vad_mode: int = 3  # WebRTC VAD aggressiveness, 3 = strictest
    barge_min_rms: int = 500  # loudness gate that keeps the bot's own echo out
    barge_min_speech_ms: int = 120
    barge_confirm_ms: int = 2500  # resume if STT has not confirmed by then
    #: Silence to wait after the closing line before hanging up.
    end_silence_ms: int = 1500
    end_silence_max_ms: int = 3000  # sound on the line can stretch the wait, never past this

    #: Debug only: save the caller's audio per call. Files are deleted after 7 days.
    debug_record_audio: bool = False
    recordings_dir: Path = PROJECT_ROOT / "data" / "recordings"
    #: Where Asterisk leaves per-call RTP statistics at hang-up (see asterisk/extensions.conf).
    rtp_stats_dir: Path = Path("/var/lib/voice-agent-rtp")

    #: Where per-call JSONL transcripts are written.
    calls_dir: Path = PROJECT_ROOT / "data" / "calls"
    #: Cached audio for fixed phrases (greeting, apology, goodbye).
    tts_cache_dir: Path = PROJECT_ROOT / "data" / "tts_cache"

    @property
    def echo_delay_s(self) -> float:
        return self.echo_delay_ms / 1000.0

    @property
    def stt_active(self) -> bool:
        return self.stt_enabled and bool(self.deepgram_api_key)

    @property
    def agent_active(self) -> bool:
        return self.agent_enabled and bool(self.deepgram_api_key) and bool(self.groq_api_key)


def read_env_file(path: Path) -> dict[str, str]:
    """Parse a simple ``KEY=VALUE`` file. Missing file -> empty dict."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _int(values: Mapping[str, str], key: str, default: int, lo: int, hi: int) -> int:
    raw = values.get(key)
    if raw is None or raw == "":
        return default
    try:
        number = int(raw)
    except ValueError:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from None
    if not lo <= number <= hi:
        raise ValueError(f"{key} must be between {lo} and {hi}, got {number}")
    return number


def _float(values: Mapping[str, str], key: str, default: float, lo: float, hi: float) -> float:
    raw = values.get(key)
    if raw is None or raw == "":
        return default
    try:
        number = float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, got {raw!r}") from None
    if not lo <= number <= hi:
        raise ValueError(f"{key} must be between {lo} and {hi}, got {number}")
    return number


def _bool(values: Mapping[str, str], key: str, default: bool) -> bool:
    raw = (values.get(key) or "").strip().lower()
    if raw == "":
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{key} must be true or false, got {raw!r}")


def load_settings(
    env_file: Path = DEFAULT_ENV_FILE,
    environ: Optional[Mapping[str, str]] = None,
) -> Settings:
    """Build :class:`Settings` from ``.env`` overlaid with the environment."""
    values = read_env_file(env_file)
    values.update(os.environ if environ is None else environ)
    calls_dir = values.get("CALLS_DIR") or ""
    return Settings(
        port=_int(values, "AUDIOSOCKET_PORT", 9092, 1024, 65535),
        log_level=(values.get("LOG_LEVEL") or "INFO").upper(),
        echo_mode=_bool(values, "ECHO_MODE", False),
        echo_delay_ms=_int(values, "ECHO_DELAY_MS", 1000, 0, 10_000),
        deepgram_api_key=(values.get("DEEPGRAM_API_KEY") or "").strip(),
        stt_enabled=_bool(values, "STT_ENABLED", True),
        deepgram_model=values.get("DEEPGRAM_MODEL") or "nova-3",
        deepgram_language=values.get("DEEPGRAM_LANGUAGE") or "en",
        deepgram_endpointing_ms=_int(values, "DEEPGRAM_ENDPOINTING_MS", 400, 10, 5000),
        deepgram_utterance_end_ms=_int(values, "DEEPGRAM_UTTERANCE_END_MS", 1000, 1000, 5000),
        agent_enabled=_bool(values, "AGENT_ENABLED", True),
        active_profile=(values.get("ACTIVE_PROFILE") or "lead_qualification").strip(),
        profiles_dir=Path(values["PROFILES_DIR"]) if values.get("PROFILES_DIR") else PROJECT_ROOT / "profiles",
        groq_api_key=(values.get("GROQ_API_KEY") or "").strip(),
        groq_model=values.get("GROQ_MODEL") or "openai/gpt-oss-120b",
        groq_eager_min_budget=_float(values, "GROQ_EAGER_MIN_BUDGET", 0.4, 0.0, 1.0),
        aura_model=values.get("AURA_MODEL") or "aura-2-thalia-en",
        flux_model=values.get("FLUX_MODEL") or "flux-general-en",
        flux_eot_threshold=_float(values, "FLUX_EOT_THRESHOLD", 0.7, 0.5, 1.0),
        flux_eager_eot_threshold=_float(values, "FLUX_EAGER_EOT_THRESHOLD", 0.5, 0.0, 0.9),
        flux_eot_timeout_ms=_int(values, "FLUX_EOT_TIMEOUT_MS", 3000, 500, 60000),
        barge_vad_enabled=_bool(values, "BARGE_VAD_ENABLED", True),
        barge_vad_mode=_int(values, "BARGE_VAD_MODE", 3, 0, 3),
        barge_min_rms=_int(values, "BARGE_MIN_RMS", 500, 0, 20000),
        barge_min_speech_ms=_int(values, "BARGE_MIN_SPEECH_MS", 120, 20, 1000),
        barge_confirm_ms=_int(values, "BARGE_CONFIRM_MS", 2500, 200, 10000),
        end_silence_ms=_int(values, "END_SILENCE_MS", 1500, 0, 15000),
        end_silence_max_ms=_int(values, "END_SILENCE_MAX_MS", 3000, 0, 15000),
        debug_record_audio=_bool(values, "DEBUG_RECORD_AUDIO", False),
        recordings_dir=Path(values["RECORDINGS_DIR"]) if values.get("RECORDINGS_DIR") else PROJECT_ROOT / "data" / "recordings",
        rtp_stats_dir=Path(values.get("RTP_STATS_DIR") or "/var/lib/voice-agent-rtp"),
        calls_dir=Path(calls_dir) if calls_dir else PROJECT_ROOT / "data" / "calls",
        tts_cache_dir=Path(values["TTS_CACHE_DIR"]) if values.get("TTS_CACHE_DIR") else PROJECT_ROOT / "data" / "tts_cache",
    )
