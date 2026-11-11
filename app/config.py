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

    #: Where per-call JSONL transcripts are written.
    calls_dir: Path = PROJECT_ROOT / "data" / "calls"

    @property
    def echo_delay_s(self) -> float:
        return self.echo_delay_ms / 1000.0

    @property
    def stt_active(self) -> bool:
        return self.stt_enabled and bool(self.deepgram_api_key)


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
        calls_dir=Path(calls_dir) if calls_dir else PROJECT_ROOT / "data" / "calls",
    )
